import ivy

from efficientnetv1 import EfficientNetV1
from utils import download_weights

import json
import os

ivy.set_torch_backend()


with open("variant_configs.json") as json_file:
    configs = json.load(json_file)

configs = configs["v1"]
base_model = configs["base_args"]
phi_values = configs["phi_values"]["b0"]

ivy_model = EfficientNetV1(
    base_model,
    phi_values,
    1000,
    # device='cpu'
)

# copy weights
weight_path = "weights/b0.pickled"
if not os.path.isfile(weight_path):
    download_weights(weight_path)

torch_v = ivy.Container.cont_from_disk_as_pickled(weight_path)
torch_list_weights = torch_v.cont_to_flat_list()

ivy_model.v.classifier.submodules.v1.b = torch_list_weights[0].to_device("gpu:0")
ivy_model.v.classifier.submodules.v1.w = torch_list_weights[1].to_device("gpu:0")
print(ivy_model.v.classifier.submodules.v1.w.dtype)
del torch_list_weights[:2]


def _copy_weights(dictionary):
    for key, value in list(dictionary.items()):
        # print(key, type(value))
        if isinstance(value, dict):
            _copy_weights(value)
        else:
            assert torch_list_weights[0].shape == value.shape
            dictionary.pop(key)
            dictionary[key] = torch_list_weights[0].to_device("gpu:0")
            del torch_list_weights[0]


for k, v in ivy_model.v.features.submodules.items():
    _copy_weights(v)

# inference

res = phi_values["resolution"]
x = ivy.random_normal(shape=(16, res, res, 3), dtype="float32")

print(ivy_model(x).shape)
