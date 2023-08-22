# global
import ivy
import torch
import urllib
import os
import copy


def _prune_keys(raw, ref, raw_keys_to_prune=[], ref_keys_to_prune=[]):
    pruned_ref = {}
    if raw_keys_to_prune:
        raw = raw.cont_prune_keys(raw_keys_to_prune)
    if ref_keys_to_prune:
        pruned_ref = ref.cont_at_keys(ref_keys_to_prune)
        ref = ref.cont_prune_keys(ref_keys_to_prune)
    return raw, ref, pruned_ref


def _map_weights(raw, ref, custom_mapping=None):
    mapping = {}
    for old_key, new_key in zip(
        raw.cont_sort_by_key().cont_to_iterator_keys(),
        ref.cont_sort_by_key().cont_to_iterator_keys(),
    ):
        new_mapping = new_key
        if custom_mapping is not None:
            new_mapping = custom_mapping(old_key, new_key)
        mapping[old_key] = new_mapping
    return mapping


def _rename_weights(raw, ref, rename_dict={}):
    renamed_ref = []
    for raw_key, ref_key in rename_dict.items():
        old_v = raw.cont_at_keys(raw_key)
        new_v = ref.cont_at_keys(ref_key)
        mapping = {}
        for old_key, new_key in zip(
            old_v.cont_sort_by_key().cont_to_iterator_keys(),
            new_v.cont_sort_by_key().cont_to_iterator_keys(),
        ):
            mapping[old_key] = new_key

        renamed_ref.append(old_v.cont_restructure(mapping, keep_orig=False))
        raw = raw.cont_prune_keys(raw_key)
        ref = ref.cont_prune_keys(ref_key)
    return raw, ref, renamed_ref


def _with_mha(raw, name="attention", delimeter="/"):
    attn_raw = raw.cont_at_keys(name)
    attn_keys = list(attn_raw.cont_to_iterator_keys())
    raw = raw.cont_prune_key_chains(attn_keys)
    attn = attn_raw.cont_to_flat_list()

    for i in range(0, len(attn), 8):
        key_start = delimeter.join(attn_keys[i].split(delimeter)[:-2])
        lin_b, lin_w, lin1_b, lin1_w, lin2_b, lin2_w, lin3_b, lin3_w = attn[i : i + 8]
        merge = ivy.shape(lin_b)[0] == ivy.shape(lin3_b)[0]

        in_b = ivy.concat((lin_b, lin1_b, lin2_b))
        raw = raw.cont_set_at_key_chain(key_start + delimeter + "out_proj_bias", lin3_b)
        raw = raw.cont_set_at_key_chain(
            key_start + delimeter + "out_proj_weights", lin3_w
        )
        raw = raw.cont_set_at_key_chain(
            key_start + delimeter + "in_proj_bias", in_b._data
        )
        if merge:
            in_w = ivy.concat((lin_w, lin1_w, lin2_w), axis=1)
            raw = raw.cont_set_at_key_chain(
                key_start + delimeter + "in_proj_weights", in_w._data
            )
        else:
            raw = raw.cont_set_at_key_chain(
                key_start + delimeter + "q_proj_weights", lin_w
            )
            raw = raw.cont_set_at_key_chain(
                key_start + delimeter + "k_proj_weights", lin1_w
            )
            raw = raw.cont_set_at_key_chain(
                key_start + delimeter + "v_proj_weights", lin2_w
            )
    return raw


def load_jax_weights(
    url,
    ref_model,
    custom_mapping=None,
    raw_keys_to_prune=[],
    ref_keys_to_prune=[],
    special_rename={},
    with_mha=False,
):
    import pickle

    ivy_jax = ivy.with_backend("jax")
    # todo: refactor this into a url load helper
    urllib.request.urlretrieve(url, filename="jax_weights.pystate")
    with open("jax_weights.pystate", "rb") as f:
        weights = pickle.loads(f.read())
    os.remove("jax_weights.pystate")

    try:
        weights = {**weights["params"], **weights["state"]}
    except KeyError:
        pass

    weights_raw = ivy.Container(
        ivy_jax.to_numpy(ivy_jax.Container(weights)).cont_to_dict()
    )
    weights_ref = ref_model.v

    if raw_keys_to_prune or ref_keys_to_prune:
        weights_raw, weights_ref, pruned_ref = _prune_keys(
            weights_raw, weights_ref, raw_keys_to_prune, ref_keys_to_prune
        )
    if special_rename:
        weights_raw, weights_ref, renamed_ref = _rename_weights(
            weights_raw, weights_ref, special_rename
        )
    if with_mha:
        weights_raw = _with_mha(weights_raw)
    mapping = _map_weights(weights_raw, weights_ref, custom_mapping=custom_mapping)

    w_clean = weights_raw.cont_restructure(mapping, keep_orig=False)
    if ref_keys_to_prune:
        w_clean = ivy.Container.cont_combine(w_clean, pruned_ref)
    if special_rename:
        w_clean = ivy.Container.cont_combine(w_clean, *renamed_ref)
    return ivy.asarray(w_clean)


def load_torch_weights(
    url,
    ref_model,
    raw_keys_to_prune=[],
    ref_keys_to_prune=[],
    custom_mapping=None,
    map_location=torch.device("cpu"),
):
    ivy_torch = ivy.with_backend("torch")
    weights = torch.hub.load_state_dict_from_url(url, map_location=map_location)
    weights_raw = ivy.Container(
        ivy_torch.to_numpy(ivy_torch.Container(weights)).cont_to_dict()
    )
    weights_raw, weights_ref, pruned_ref = _prune_keys(
        weights_raw, ref_model.v, raw_keys_to_prune, ref_keys_to_prune
    )
    mapping = _map_weights(weights_raw, weights_ref, custom_mapping=custom_mapping)
    w_clean = weights_raw.cont_restructure(mapping, keep_orig=False)
    if ref_keys_to_prune:
        w_clean = ivy.Container.cont_combine(w_clean, pruned_ref)
    return ivy.asarray(w_clean)


def _unflatten_set(container, name, to_set, split_on="__"):
    splits = name.split(split_on)
    cont = container
    for sp in splits[:-1]:
        cont = cont.setdefault(sp, {})
    cont[splits[-1]] = to_set


def load_transformers_weights(hf_repo, model, map_fn, split_on="__"):
    from transformers import AutoModel

    base = AutoModel.from_pretrained(hf_repo)
    ref_weights = base.state_dict()
    ivy_torch = ivy.with_backend("torch")
    ref_weights = ivy.Container(
        ivy_torch.to_numpy(ivy_torch.Container(ref_weights)).cont_to_dict()
    )
    old_mapping = copy.deepcopy(model.v)
    param_names = old_mapping.cont_flatten_key_chains().keys()
    mapping_list = map(lambda x: map_fn(x), param_names)
    mapping = dict(zip(param_names, mapping_list))
    for old_name, ref_name in mapping.items():
        to_set = ivy.asarray(ref_weights[ref_name])
        _unflatten_set(old_mapping, old_name, to_set, split_on)
    return old_mapping
