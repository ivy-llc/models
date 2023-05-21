import ivy
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def _permute_4d_weights(dictionary: ivy.Container, permute_fn, bias_reshape) -> None:
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
            if (
                set(value.keys()) >= set(["weight", "bias"])
                and len(value["weight"].shape) == 4
            ):
                _permute_4d_weights(value, permute_fn, True)
            else:
                _permute_4d_weights(value, permute_fn, False)
        else:
            if key.startswith("weight") and len(value.shape) == 4:
                new_value = {"weight": _reshape_weights(dictionary.pop(key))}
                dictionary[key] = ivy.Container(new_value)
            elif key.startswith("num_batches_tracked"):
                dictionary.pop(key)
            elif key.startswith("bias") and bias_reshape:
                new_value = {"bias": _reshape_bias(dictionary.pop(key))}
                dictionary[key] = ivy.Container(new_value)


def _reshape_bias(x) -> None:
    x = ivy.reshape(x, shape=(1, 1, 1, x.shape[0]))
    return x


def _reshape_weights(x) -> None:
    x = ivy.reshape(x, shape=(x.shape[2], x.shape[3], x.shape[1], x.shape[0]))
    # comment only the following if flow when using conv with groups instead of dwconv
    if x.shape[-2] == 1:
        return ivy.squeeze(x, axis=-2)
    if x.shape[-1] == 1:
        return ivy.squeeze(x, axis=-1)
    return x


def download_weights(path):
    torch_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    torch_v = ivy.Container(torch_model.state_dict())
    _permute_4d_weights(torch_v, _reshape_weights, None)
    torch_v.cont_to_disk_as_pickled(path)
