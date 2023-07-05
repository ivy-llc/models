# global
import ivy
import torch


def _prune_keys(raw, ref, raw_keys_to_prune=[], ref_keys_to_prune=[]):
    if raw_keys_to_prune != []:
        raw = raw.cont_prune_keys(raw_keys_to_prune)
    if ref_keys_to_prune != []:
        ref = ref.cont_prune_keys(ref_keys_to_prune)
    return raw, ref


def _map_weights(raw, ref, custom_mapping=None):
    mapping = {}
    for old_key, new_key in zip(
        raw.cont_sort_by_key().cont_to_iterator_keys(),
        ref.cont_sort_by_key().cont_to_iterator_keys(),
    ):
        new_mapping = new_key
        if custom_mapping is not None:
            new_mapping = custom_mapping(old_key, new_key)
            if new_mapping is None:
                continue
        mapping[old_key] = new_mapping
    return mapping


def load_torch_weights(
    url,
    ref_model,
    raw_keys_to_prune=[],
    ref_keys_to_prune=[],
    custom_mapping=None,
    map_location=torch.device("cpu"),
):
    ivy.set_backend("torch")
    weights = torch.hub.load_state_dict_from_url(url, map_location=map_location)

    weights_raw = ivy.to_numpy(ivy.Container(weights))
    weights_raw, weights_ref = _prune_keys(
        weights_raw, ref_model.v, raw_keys_to_prune, ref_keys_to_prune
    )
    mapping = _map_weights(weights_raw, weights_ref, custom_mapping=custom_mapping)

    ivy.previous_backend()
    w_clean = weights_raw.cont_restructure(mapping, keep_orig=False)
    return ivy.asarray(w_clean)


def load_jax_weights(
    url, ref_model, custom_mapping=None, raw_keys_to_prune=[], ref_keys_to_prune=[]
):
    import urllib.request
    import os
    import pickle

    ivy.set_backend("jax")
    urllib.request.urlretrieve(url, filename="jax_weights.pystate")
    with open("jax_weights.pystate", "rb") as f:
        weights = pickle.loads(f.read())
    os.remove("jax_weights.pystate")

    try:
        weights = {**weights["params"], **weights["state"]}
    except KeyError:
        pass

    weights_raw = ivy.to_numpy(ivy.Container(weights))
    weights_raw, weights_ref = _prune_keys(
        weights_raw, ref_model.v, raw_keys_to_prune, ref_keys_to_prune
    )
    mapping = _map_weights(weights_raw, weights_ref, custom_mapping=custom_mapping)

    ivy.previous_backend()
    w_clean = weights_raw.cont_restructure(mapping, keep_orig=False)
    return ivy.asarray(w_clean)


def map_cont_weights(model_weights: ivy.Container, pretrained_weights: ivy.Container, mapping_table: dict, copy=False):
    """
    Maps pretrained_weights to model_weights.
    Also apply an optional transformation function on 'pretrained_weights' before assigning the result to the coresponding key_chain in model_weights.
    Parameters
    ----------
    model_weights
            Ivy model weights
    pretrained_weights
            Pretrained torch weights to load in ivy models 
    mapping_table
            A dictionary mapping all the flatten keys of model_weights to a dictionnary with a 'context' key representing a key_chain to a tensor in pretrained_weights
            and an optional 'func' key representing an optional function to apply to the pretrained weight before assigning the result to the ivy model. 
            E.g. {'transformer/submodules/v0/attn/to_q/w': {'context':'transformer/0/attn/in_pro_weights', 'func': lambda x: x[:x.shape[0]//3},}
    """
    if copy:
        model_weights = model_weights.cont_deep_copy()

    for key, val in mapping_table.items():
        temp_weight = pretrained_weights[val['context']]
        temp_val = val['func'](temp_weight) if 'func' in val else temp_weight
        if list(temp_val.shape) != list(model_weights[key].shape):
            raise RuntimeError(f"The shape of the new tensor at {key} is {temp_val.shape}, but expected {model_weights[key].shape}")
        model_weights[key] = temp_val
    return ivy.asarray(model_weights)


def test_weights_closeness(model_weights: ivy.Module, pretrained_weights: ivy.Container, mapping_table: dict, atol=0.0001):
    for key, val in mapping_table.items():
        temp_weight = pretrained_weights[val['context']]
        temp_val = val['func'](temp_weight) if 'func' in val else temp_weight

        if not ivy.allclose(model_weights[key], temp_val, atol=atol).item():
            raise RuntimeError(f"Closeness test failed for atol={atol}. Model weight's key {key} | Pretrained weight's key {val['context']}")
