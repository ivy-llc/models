# global
import ivy


def _prune_keys(raw, ref, raw_keys_to_prune=[], ref_keys_to_prune=[]):
    if raw_keys_to_prune != []:
        for kc in raw_keys_to_prune:
            raw = raw.cont_prune_key_from_key_chains(containing=kc)
    if ref_keys_to_prune != []:
        for kc in ref_keys_to_prune:
            ref = ref.cont_prune_key_from_key_chains(containing=kc)
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


def load_torch_weights(url, ref_model, custom_mapping=None):
    import torch

    ivy.set_backend("torch")
    weights = torch.hub.load_state_dict_from_url(url)
    weights_raw = ivy.to_numpy(ivy.Container(weights))
    mapping = _map_weights(weights_raw, ref_model.v, custom_mapping=custom_mapping)

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


def _permute_4d_weights(dictionary: ivy.Container, permute_fn) -> None:
    """
    Recursively permutes 4D weights in the dictionary
                using the provided permutation function.

    Args::
        dictionary (ivy.Container): The dictionary to process.
        permute_fn: The permutation function to apply to the weights.

    Returns::
        None
    """
    for key, value in list(dictionary.items()):
        if isinstance(value, dict):
            _permute_4d_weights(value, permute_fn)
        else:
            if key.startswith("weight") and len(value.shape) == 4:
                new_value = {"weight": permute_fn(dictionary.pop(key))}
                dictionary[key] = ivy.Container(new_value)


def _rename_w_b(dictionary: ivy.Container) -> None:
    """
    Recursively renames the 'weight' and 'bias' keys in the dictionary.

    Args::
        dictionary (ivy.Container): The dictionary to process.

    Returns::
        None
    """
    keys = list(dictionary.keys())
    for key in keys:
        value = dictionary[key]
        if isinstance(value, dict):
            _rename_w_b(value)
        if key.startswith("weight"):
            dictionary["w" + key[6:]] = dictionary.pop(key)
        elif key.startswith("bias"):
            dictionary["b" + key[4:]] = dictionary.pop(key)


def _custom_resnet_renaming(dictionary: ivy.Container) -> None:
    """
    Recursively renames keys in the dictionary
    according to custom ResNet naming conventions.

    Args::
        dictionary (ivy.Container): The dictionary to process.

    Returns::
        None
    """
    for key, value in list(dictionary.items()):
        if isinstance(value, dict):
            _custom_resnet_renaming(value)

        if key in ["0", "1"]:
            new_key = "v" + key
            dictionary[new_key] = dictionary.pop(key)

        if key.startswith("layer"):
            dictionary[key] = ivy.Container({"layers": value})
