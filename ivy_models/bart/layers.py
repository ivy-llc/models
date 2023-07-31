import ivy
from typing import Optional, Tuple
from .config_bart import BartConfig
from .activations import ACT2FN


class BartLearnedPositionalEmbedding(ivy.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def _forward(self, input_ids: ivy.Array, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""

        bsz, seq_len = input_ids.shape[:2]
        positions = ivy.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=ivy.int64,
            device=self.weight.device,
        )
        ivy.expand(positions, (bsz, -1))

        return super()._forward(positions + self.offset)


class BartAttention(ivy.Module):
    def init(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = ivy.Linear(embed_dim, embed_dim, with_bias=bias)
        self.v_proj = ivy.Linear(embed_dim, embed_dim, with_bias=bias)
        self.q_proj = ivy.Linear(embed_dim, embed_dim, with_bias=bias)
        self.out_proj = ivy.Linear(embed_dim, embed_dim, with_bias=bias)

    def _shape(self, tensor: ivy.Array, seq_len: int, bsz: int):
        return ivy.swapaxes(
            ivy.reshape(tensor, (bsz, seq_len, self.num_heads, self.head_dim)), 1, 2
        )

    def _forward(
        self,
        hidden_states: ivy.Array,
        key_value_states: Optional[ivy.Array] = None,
        past_key_value: Optional[Tuple[ivy.Array]] = None,
        attention_mask: Optional[ivy.Array] = None,
        layer_head_mask: Optional[ivy.Array] = None,
        output_attentions: bool = False,
    ) -> Tuple[ivy.Array, Optional[ivy.Array], Optional[Tuple[ivy.Array]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ivy.concat([past_key_value[0], key_states], dim=2)
            value_states = ivy.concat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(ivy.Array, ivy.Array) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(ivy.Array, ivy.Array) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = ivy.reshape(self._shape(query_states, tgt_len, bsz), *proj_shape)
        key_states = ivy.reshape(key_states, *proj_shape)
        value_states = ivy.reshape(value_states, *proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ivy.matmul(query_states, ivy.swapaxes(key_states, 1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = (
                attn_weights.shape[bsz, self.num_heads, tgt_len, src_len]
                + attention_mask
            )
            attn_weights = attn_weights.shape[bsz * self.num_heads, tgt_len, src_len]

        attn_weights = ivy.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.shape}"
                )
            attn_weights = (
                layer_head_mask.shape[1, -1, 1, 1]
                * attn_weights.shape[bsz, self.num_heads, tgt_len, src_len]
            )
            attn_weights = attn_weights.shape[bsz * self.num_heads, tgt_len, src_len]

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.shape[
                bsz, self.num_heads, tgt_len, src_len
            ]
            attn_weights = attn_weights_reshaped.shape[
                bsz * self.num_heads, tgt_len, src_len
            ]
        else:
            attn_weights_reshaped = None

        attn_probs = ivy.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = ivy.matmul(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.shape[bsz, self.num_heads, tgt_len, self.head_dim]
        attn_output = ivy.swapaxes(attn_output, 1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = ivy.reshape(attn_output, (bsz, tgt_len, self.embed_dim))

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(ivy.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = ivy.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = ivy.LayerNorm(self.embed_dim)
        self.fc1 = ivy.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = ivy.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = ivy.LayerNorm(self.embed_dim)

    def _forward(
        self,
        hidden_states: ivy.Array,
        attention_mask: Optional[ivy.Array] = None,
        encoder_hidden_states: Optional[ivy.Array] = None,
        encoder_attention_mask: Optional[ivy.Array] = None,
        layer_head_mask: Optional[ivy.Array] = None,
        cross_attn_layer_head_mask: Optional[ivy.Array] = None,
        past_key_value: Optional[Tuple[ivy.Array]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[ivy.Array, Optional[Tuple[ivy.Array, ivy.Array]]]:
        """
        Args:
            hidden_states (`ivy.Array`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ivy.Array`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`ivy.Array`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`ivy.Array`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`ivy.Array`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`ivy.Array`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(ivy.Array)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = ivy.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = ivy.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

            # Fully Connected
            residual = hidden_states
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            hidden_states = ivy.dropout(
                hidden_states, p=self.activation_dropout, training=self.training
            )
            hidden_states = self.fc2(hidden_states)
            hidden_states = ivy.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.final_layer_norm(hidden_states)

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights, cross_attn_weights)

            if use_cache:
                outputs += (present_key_value,)

            return outputs


class BartDecoderLayer(ivy.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = ivy.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = ivy.LayerNorm(self.embed_dim)
        self.fc1 = ivy.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = ivy.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = ivy.LayerNorm(self.embed_dim)

    def _forward(
        self,
        hidden_states: ivy.Array,
        attention_mask: Optional[ivy.Array] = None,
        encoder_hidden_states: Optional[ivy.Array] = None,
        encoder_attention_mask: Optional[ivy.Array] = None,
        layer_head_mask: Optional[ivy.Array] = None,
        cross_attn_layer_head_mask: Optional[ivy.Array] = None,
        past_key_value: Optional[Tuple[ivy.Array]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[ivy.Array, Optional[Tuple[ivy.Array, ivy.Array]]]:
        """
        Args:
            hidden_states (`ivy.Array`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ivy.Array`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`ivy.Array`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`ivy.Array`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`ivy.Array`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`ivy.Array`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(ivy.Array)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = ivy.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = ivy.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ivy.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = ivy.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartClassificationHead(ivy.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = ivy.Linear(input_dim, inner_dim)
        self.dropout = ivy.Dropout(p=pooler_dropout)
        self.out_proj = ivy.Linear(inner_dim, num_classes)

    def _forward(self, hidden_states: ivy.Array) -> ivy.Array:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ivy.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
