from ivy_models.base import BaseSpec


class BartConfig(BaseSpec):
    model_type: str = "bart"
    problem_type: str = "multi_label_classification"
    keys_to_ignore_at_inference: list = ["past_key_values"]
    attribute_map: dict = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }

    vocab_size: int = 50265
    max_position_embeddings: int = 1024
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0
    activation_function: str = "gelu"
    d_model: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    init_std: float = 0.02
    classifier_dropout: float = 0.0
    scale_embedding: bool = False
    use_cache: bool = True
    num_labels: int = 3
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    is_encoder_decoder: bool = True
    is_decoder: bool = False
    decoder_start_token_id: int = 2
    forced_eos_token_id: int = 2
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = False

    def get(self, *attr_names):
        new_dict = {}
        for name in attr_names:
            new_dict[name] = getattr(self, name)
        return new_dict
