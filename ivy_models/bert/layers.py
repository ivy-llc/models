import ivy
import math


class BertEmbedding(ivy.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        type_vocab_size=2,
        pad_token_id=None,
        embd_drop_rate=0.1,
        layer_norm_eps=1e-5,
        position_embedding_type="absolute",
        v=None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.type_token_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.padding_idx = pad_token_id
        self.drop_rate = embd_drop_rate
        self.position_type_embd = position_embedding_type
        self.layer_norm_eps = layer_norm_eps
        super(BertEmbedding, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.word_embeddings = ivy.Embedding(
            self.vocab_size, self.hidden_size, self.padding_idx
        )
        self.position_embeddings = ivy.Embedding(
            self.max_position_embeddings, self.hidden_size
        )
        self.token_type_embeddings = ivy.Embedding(
            self.type_token_size, self.hidden_size
        )
        self.dropout = ivy.Dropout(self.drop_rate)
        self.LayerNorm = ivy.LayerNorm([self.hidden_size], eps=self.layer_norm_eps)

    def _forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        past_key_values_length: int = 0,
    ):
        input_shape = input_ids.shape
        seq_length = input_shape[1]

        if position_ids is None:
            pos_ids = ivy.expand_dims(ivy.arange(self.max_position_embeddings), axis=0)
            position_ids = pos_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        if token_type_ids is None:
            token_type_ids = ivy.zeros(
                (1, self.max_position_embeddings), dtype=ivy.int32
            )
            buffered_token_type_ids = token_type_ids[:, :seq_length]
            token_type_ids = buffered_token_type_ids.expand(
                (input_shape[0], seq_length)
            )

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_type_embd == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(ivy.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        max_position_embeddings,
        position_embedding_type=None,
        attn_drop_rate=0.1,
        is_decoder=False,
        v=None,
    ):
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size})"
                f" is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = hidden_size
        self.attn_drop_rate = attn_drop_rate
        self.position_type_embd = (
            position_embedding_type
            if position_embedding_type is not None
            else "absolute"
        )
        self.is_decoder = is_decoder
        self.max_position_embeddings = max_position_embeddings
        super(BertSelfAttention, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.query = ivy.Linear(self.hidden_size, self.all_head_size)
        self.key = ivy.Linear(self.hidden_size, self.all_head_size)
        self.value = ivy.Linear(self.hidden_size, self.all_head_size)
        self.dropout = ivy.Dropout(self.attn_drop_rate)

    def transpose_for_scores(self, x: ivy.Array):
        # transpose the hidden_states from (bs, seq_len, hidden_size)
        # - > (bs, seq_len, num_heads, head_size)
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.reshape(new_x_shape)
        return x.permute_dims((0, 2, 1, 3))

    def _forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = ivy.concat([past_key_value[0], key_layer], dim=2)
            value_layer = ivy.concat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        # scaled Dot product
        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = ivy.matmul(query_layer, key_layer.permute_dims((0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Masking
        if attention_mask is not None:
            shape = ivy.shape(attention_mask)
            attention_mask = ivy.astype(attention_mask, bool)
            if len(shape) == 2:
                attention_mask = ivy.expand_dims(attention_mask, axis=(1, -1))
            elif len(shape) == 3:
                attention_mask = ivy.expand_dims(attention_mask, axis=-1)

            attention_scores = ivy.where(attention_mask, attention_scores, -ivy.inf)

        # Normalize the attention scores to probabilities.
        attention_probs = ivy.softmax(attention_scores, axis=-1)
        # solving the softmax nan number problem
        attention_probs = ivy.nan_to_num(attention_probs, nan=0.0)

        attention_probs = self.dropout(attention_probs)

        context_layer = ivy.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute_dims((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertAttention(ivy.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        max_position_embeddings,
        position_embedding_type=None,
        attn_drop_rate=0.1,
        hidden_dropout=0.1,
        layer_norm_eps=1e-5,
        is_decoder=False,
        v=None,
    ):
        self.hidden_size = hidden_size
        self.attn_drop_rate = attn_drop_rate
        self.position_type_embd = (
            position_embedding_type
            if position_embedding_type is not None
            else "absolute"
        )
        self.hidden_dropout = hidden_dropout
        self.is_decoder = is_decoder
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        super(BertAttention, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.self = BertSelfAttention(
            self.hidden_size,
            self.num_attention_heads,
            self.max_position_embeddings,
            self.position_type_embd,
            self.attn_drop_rate,
            self.is_decoder,
        )
        self.dense = ivy.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = ivy.LayerNorm([self.hidden_size], eps=self.layer_norm_eps)
        self.dropout = ivy.Dropout(self.hidden_dropout)

    def _forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        outputs = self.self(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )

        out = self.dense(outputs[0])
        out = self.LayerNorm(self.dropout(out) + hidden_states)
        outputs = (out,) + outputs[1:]
        return outputs


class BertFeedForward(ivy.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act,
        layer_norm_eps=1e-5,
        ffd_drop=0.1,
        v=None,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = getattr(ivy, hidden_act.lower())
        self.ffd_drop = ffd_drop
        self.layer_norm_eps = layer_norm_eps
        super(BertFeedForward, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.dense1 = ivy.Linear(self.hidden_size, self.intermediate_size)
        self.dense2 = ivy.Linear(self.intermediate_size, self.hidden_size)
        self.LayerNorm = ivy.LayerNorm([self.hidden_size], eps=self.layer_norm_eps)
        self.dropout = ivy.Dropout(self.ffd_drop)

    def _forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.hidden_act(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(residual + hidden_states)
