import re
import warnings
from typing import Union, List
from pkg_resources import packaging


import ivy
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

import ivy_models
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        lambda x: ivy.array(x.numpy()),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


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

def get_ivy_weights(model_weights: ivy.Container, state_dict: dict) -> ivy.Container:
    """
    This function returns the ivy model weights with the appropriate Container structure, filled with the pretrained weights.
    It creates a mapping table that's used to convert the reference torch weights to the ivy model 
    The ivy model has way more weight tensors than the reference torch model, mainly due to the MultiHeadAttention module.
    So some manual mapping was needed.
    """
    
    ivy.set_backend("torch")

    mapping_table = {}
    pretrained_weights = ivy.to_numpy(ivy.Container(state_dict).astype(ivy.float32))

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
                f"{ivy_root}attn/out_proj_weights" : {'context': f"{torch_root}attn/out_proj/weight"},
                f"{ivy_root}attn/out_proj_bias" : {'context': f"{torch_root}attn/out_proj/bias"},
                f"{ivy_root}attn/in_proj_weights" : {'context': f"{torch_root}attn/in_proj_weight"},
                f"{ivy_root}attn/in_proj_bias" : {'context': f"{torch_root}attn/in_proj_bias"},
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
    ivy.previous_backend()
    clean_weights = ivy_models.helpers.map_cont_weights(model_weights, pretrained_weights, mapping_table)

    return clean_weights

def load_clip_state_dict(name: str):
    if name in _MODELS:
        url = _MODELS[name]
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu').state_dict()
        return state_dict
    else:
        raise ValueError(f"Model '{name}' not found; available models = {available_models()}")


def get_processors(model):
    """
    Returns the text tokenizer and the approiate image transform depending on the model variant.
    """
    return tokenize, _transform(model.visual.input_resolution)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return ivy.array(result.numpy())

