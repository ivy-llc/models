import ivy

from efficientnetv1 import EfficientNetV1
from utils import download_weights

import json
import os

# ivy.set_tensorflow_backend()
# ivy.set_jax_backend()
ivy.set_torch_backend()

device = "gpu:0"

with open("variant_configs.json") as json_file:
    configs = json.load(json_file)

configs = configs["v1"]
base_model = configs["base_args"]
phi_values = configs["phi_values"]["b0"]

ivy_model = EfficientNetV1(base_model, phi_values, 1000, device=device, training=False)

# copy weights
weight_path = "weights/b0.pickled"
if not os.path.isfile(weight_path):
    download_weights(weight_path)

torch_v = ivy.Container.cont_from_disk_as_pickled(weight_path)
torch_list_weights = torch_v.cont_to_flat_list()

ivy_model.v.classifier.submodules.v1.b = ivy.Array(
    torch_list_weights[0].detach().cpu().numpy()
).to_device(device)
ivy_model.v.classifier.submodules.v1.w = ivy.Array(
    torch_list_weights[1].detach().cpu().numpy()
).to_device(device)
del torch_list_weights[:2]


def _copy_weights(dictionary):
    for key, value in list(dictionary.items()):
        # print(key, type(value))
        if isinstance(value, dict):
            _copy_weights(value)
        else:
            assert torch_list_weights[0].shape == value.shape
            dictionary.pop(key)
            dictionary[key] = ivy.Array(
                torch_list_weights[0].detach().cpu().numpy()
            ).to_device(device)
            del torch_list_weights[0]


for k, v in ivy_model.v.features.submodules.items():
    _copy_weights(v)

# inference

res = phi_values["resolution"]
x = ivy.random_normal(shape=(1, res, res, 3), dtype="float32", device=device)

print(ivy_model(x).shape)
