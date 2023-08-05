import ivy
from ivy_models.bert.layers import BertEmbedding


class RobertaEmbeddings(BertEmbedding):
    """Same as Bert Embedding with tiny change in the positional indexing."""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        type_vocab_size=1,
        pad_token_id=None,
        embd_drop_rate=0.1,
        layer_norm_eps=1e-5,
        position_embedding_type="absolute",
        v=None,
    ):
        super(RobertaEmbeddings, self).__init__(
            vocab_size,
            hidden_size,
            max_position_embeddings,
            type_vocab_size,
            pad_token_id,
            embd_drop_rate,
            layer_norm_eps,
            position_embedding_type,
            v,
        )
        self.padding_idx = 1

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
            position_ids = ivy.expand_dims(
                ivy.arange(self.padding_idx + 1, seq_length + self.padding_idx), axis=0
            )
            position_ids = position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]
        return super(RobertaEmbeddings, self)._forward(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values_length=past_key_values_length,
        )
