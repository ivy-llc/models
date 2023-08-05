from ivy_models.helpers import load_transformers_weights
from ivy_models.bert import BertConfig, BertModel
from .layers import RobertaEmbeddings


class RobertaModel(BertModel):
    def __init__(self, config: BertConfig, pooler_out=False):
        super(RobertaModel, self).__init__(config, pooler_out=pooler_out)

    @classmethod
    def get_spec_class(self):
        return BertConfig

    def _build(self, *args, **kwargs):
        self.embeddings = RobertaEmbeddings(**self.config.get_embd_attrs())
        super(RobertaModel, self)._build(*args, **kwargs)

    def _forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
    ):
        if input_ids[:, 0].sum().item() != 0:
            print("NOT ALLOWED")
        return super(RobertaModel, self)._forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


def _roberta_weights_mapping(name):
    key_map = [(f"__v{i}__", f"__{j}__") for i, j in zip(range(12), range(12))]
    key_map = key_map + [
        ("attention__dense", "attention.output.dense"),
        ("attention__LayerNorm", "attention.output.LayerNorm"),
    ]
    key_map = key_map + [
        ("ffd__dense1", "intermediate.dense"),
        ("ffd__dense2", "output.dense"),
        ("ffd__LayerNorm", "output.LayerNorm"),
    ]
    name = name.replace("__w", ".weight").replace("__b", ".bias")
    name = (
        name.replace("biasias", "bias")
        .replace("weighteight", "weight")
        .replace(".weightord", ".word")
    )
    for ref, new in key_map:
        name = name.replace(ref, new)
    name = name.replace("__", ".")
    return name


def roberta_base(pretrained=True):
    # instantiate the hyperparameters same as bert
    # set the dropout rate to 0.0 to avoid stochasticity in the output

    config = BertConfig(
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.0,
        attn_drop_rate=0.0,
        max_position_embeddings=514,
        type_vocab_size=1,
    )
    model = RobertaModel(config, pooler_out=True)
    if pretrained:
        w_clean = load_transformers_weights(
            "roberta-base", model, _roberta_weights_mapping
        )
        model.v = w_clean
    return model
