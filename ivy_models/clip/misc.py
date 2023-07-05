import re

import ivy
import ivy_models
from .model import CLIP


def get_model_args(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    return embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, \
        transformer_width, transformer_heads, transformer_layers

def get_ivy_weights(model_weights: ivy.Container, state_dict: dict) -> CLIP:
    """
    This function returns the ivy model weights with the appropriate Container structure, filled with the pretrained weights.
    It creates a mapping table that's used to convert the reference torch weights to the ivy model 
    The ivy model has way more weight tensors than the reference torch model, mainly due to the MultiHeadAttention module.
    So some manual mapping was needed.
    """
    
    mapping_table = {}
    pretrained_weights = ivy.asarray(ivy.Container(state_dict).astype(float))

    pretrained_key_chains = pretrained_weights.cont_sort_by_key().cont_all_key_chains()
    model_key_chains = model_weights.cont_all_key_chains()
    visited = []

    for key_chain in pretrained_key_chains:
        # Automatically includes /submodules/v{number} for elligible layers
        ivy_key = key_chain
        pattern = r".*?/(\d+)/.*"
        if re.match(pattern, key_chain):
            res = re.search(pattern, key_chain)
            number = res.group(1)
            ivy_key = key_chain[:res.start(1)] + f"submodules/v{number}" + key_chain[res.end(1):]

        # Skip modules whose submodules are already added to avoid redundant computations
        if [True for root in visited if root in ivy_key]:
            continue

        # Add key chains with exact match in the two weights containers
        if key_chain in model_key_chains:
            mapping_table[key_chain] = {'context': key_chain}
            continue

        # Modified Resnet
        # Conv weight and bias
        pattern = r".*/conv(\d+)/weight"
        if re.match(pattern, ivy_key):
            ivy_root = ivy_key[:re.search(pattern, ivy_key).span()[1]-6]
            mapping_table[ivy_root + "w"] = {'context': key_chain, 'func': lambda x: ivy.einops_rearrange(x, "o c h w -> h w c o ")}
        # Attention pooling
        pattern = r".*attnpool/.*/.*"
        if re.match(pattern, ivy_key):
            ivy_root = ivy_key[:ivy_key.rfind('/')+1]
            torch_root = key_chain[:key_chain.rfind('/')+1]
            mapping_table[ivy_root + "w"] = {'context': torch_root + "weight"}
            mapping_table[ivy_root + "b"] = {'context': torch_root + "bias"}
        # BatchNorms
        pattern = r".*/bn(\d+)/.*"
        if re.match(pattern, ivy_key):
            ivy_root = ivy_key[:ivy_key.rfind('/')+1]
            torch_root = key_chain[:key_chain.rfind('/')+1]
            temp = {
                f"{ivy_root}w": {'context': f"{torch_root}weight"},
                f"{ivy_root}b": {'context': f"{torch_root}bias"},
                f"{ivy_root}running_mean": {'context': f"{torch_root}running_mean"},
                f"{ivy_root}running_var": {'context': f"{torch_root}running_var"},
            }
            mapping_table.update(temp)
        # Downsample
        pattern = r".*/downsample/(\d+)/.*"
        if re.match(pattern, ivy_key):
            ivy_root = ivy_key[:ivy_key.rfind('/')-1]
            torch_root = key_chain[:key_chain.rfind('/')-1]
            # torch starts with 0, but ivy skips the layer with non learnable params so we have v1
            temp = {
                f"{ivy_root}submodules/v1/w": {'context': f"{torch_root}0/weight", 'func': lambda x: ivy.einops_rearrange(x, "o c h w -> h w c o ")},
                f"{ivy_root}submodules/v2/w": {'context': f"{torch_root}1/weight"},
                f"{ivy_root}submodules/v2/b": {'context': f"{torch_root}1/bias"},
                f"{ivy_root}submodules/v2/running_mean": {'context': f"{torch_root}1/running_mean"},
                f"{ivy_root}submodules/v2/running_var": {'context': f"{torch_root}1/running_var"},
            }
            mapping_table.update(temp)

        # Text Transformer and ViT
        pattern = r".*transformer/.*/v(\d+)/attn/"
        if re.match(pattern, ivy_key):
            ivy_root = ivy_key[:ivy_key.rfind('/attn/')+1]
            torch_root = key_chain[:key_chain.rfind('/attn/')+1]

            temp = {
                f"{ivy_root}attn/to_kv/k/b": {'context': f"{torch_root}attn/in_proj_bias", 'func': lambda x: x[x.shape[0]//3:(x.shape[0]//3)*2]},
                f"{ivy_root}attn/to_kv/k/w": {'context': f"{torch_root}attn/in_proj_weight", 'func': lambda x: x[x.shape[0]//3:(x.shape[0]//3)*2]},
                f"{ivy_root}attn/to_kv/v/b": {'context': f"{torch_root}attn/in_proj_bias", 'func': lambda x: x[-x.shape[0]//3:]},
                f"{ivy_root}attn/to_kv/v/w": {'context': f"{torch_root}attn/in_proj_weight", 'func': lambda x: x[-x.shape[0]//3:]},
                f"{ivy_root}attn/to_out/submodules/v0/b": {'context': f"{torch_root}attn/out_proj/bias"},
                f"{ivy_root}attn/to_out/submodules/v0/w": {'context': f"{torch_root}attn/out_proj/weight"},
                f"{ivy_root}attn/to_q/b": {'context': f"{torch_root}attn/in_proj_bias", 'func': lambda x: x[:x.shape[0]//3]},
                f"{ivy_root}attn/to_q/w": {'context': f"{torch_root}attn/in_proj_weight", 'func': lambda x: x[:x.shape[0]//3]},
                f"{ivy_root}ln_1/bias": {'context': f"{torch_root}ln_1/bias"},
                f"{ivy_root}ln_1/weight": {'context': f"{torch_root}ln_1/weight"},
                f"{ivy_root}ln_2/bias": {'context': f"{torch_root}ln_2/bias"},
                f"{ivy_root}ln_2/weight": {'context': f"{torch_root}ln_2/weight"},
                f"{ivy_root}mlp/submodules/v0/b": {'context': f"{torch_root}mlp/c_fc/bias"},
                f"{ivy_root}mlp/submodules/v0/w": {'context': f"{torch_root}mlp/c_fc/weight"},
                f"{ivy_root}mlp/submodules/v2/b": {'context': f"{torch_root}mlp/c_proj/bias"},
                f"{ivy_root}mlp/submodules/v2/w": {'context': f"{torch_root}mlp/c_proj/weight"},
            }
            mapping_table.update(temp)
    
    assert(len(mapping_table) == len(model_key_chains))

    # Load weights and do some minor chekings
    clean_weights = ivy_models.helpers.map_cont_weights(model_weights, pretrained_weights, mapping_table)
    # ivy_models.helpers.test_weights_closeness(clean_weights, pretrained_weights, mapping_table, atol=0.00002)

    return clean_weights
