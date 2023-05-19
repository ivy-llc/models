import ivy
from torchvision import transforms

from efficientnetv1 import EfficientNetV1
from utils import download_weights
import json
from PIL import Image
import os


device = "cpu"

with open("variant_configs.json") as json_file:
    configs = json.load(json_file)

configs = configs["v1"]
base_model = configs["base_args"]
phi_values = configs["phi_values"]["b0"]


def create_model():
    ivy_model = EfficientNetV1(
        base_model, phi_values, 1000, device=device, training=False
    )

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
                assert (
                    torch_list_weights[0].shape == value.shape
                ), f"{torch_list_weights[0].shape}, {value.shape}"
                dictionary.pop(key)
                dictionary[key] = ivy.Array(
                    torch_list_weights[0].detach().cpu().numpy()
                ).to_device(device)
                del torch_list_weights[0]

    for k, v in ivy_model.v.features.submodules.items():
        _copy_weights(v)
    return ivy_model


# inference

res = phi_values["resolution"]
# image_path = "images/dog.jpeg"
image_path = "images/ILSVRC2012_test_00000007.jpeg"
# image_path = 'images/ILSVRC2012_test_00000030.jpeg'
# Define the transformation pipeline
preprocess = transforms.Compose(
    [
        transforms.Resize(
            256, interpolation=transforms.InterpolationMode.BICUBIC
        ),  # Resize the image to a square of size 256x256
        transforms.CenterCrop(res),  # Crop the center portion of the image to 224x224
        transforms.ToTensor(),  # Convert the PIL image to a tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize the image
    ]
)


def get_processed_image(image_path):
    x = Image.open(image_path)
    x = preprocess(x).unsqueeze(0)
    x = x.detach().cpu().numpy()
    x = x.reshape((1, res, res, 3))
    return x


print(image_path)

ivy.set_torch_backend()

input_img = ivy.Array(get_processed_image(image_path)).to_device(device)
ivy_model = create_model()
output = ivy.softmax(ivy_model(input_img))
print("torch", output[0].argmax(), output[0].sort()[-5:])


ivy.set_jax_backend()

input_img = ivy.Array(get_processed_image(image_path)).to_device(device)
ivy_model = create_model()
output = ivy.softmax(ivy_model(input_img))
print("jax", output[0].argmax(), output[0].sort()[-5:])


ivy.set_tensorflow_backend()

input_img = ivy.Array(get_processed_image(image_path)).to_device(device)
ivy_model = create_model()
output = ivy.softmax(ivy_model(input_img))
print("tensorflow", output[0].argmax(), output[0].sort()[-5:])
