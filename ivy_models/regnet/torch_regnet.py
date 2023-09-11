import torch

# wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# wget https://raw.githubusercontent.com/unifyai/models/master/images/cat.jpg
filename = "cat.jpg"
# Preprocess torch image
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
torch_img = Image.open(filename)
torch_img = preprocess(torch_img)
torch_img = torch.unsqueeze(torch_img, 0)


from torchvision.models import *

VARIANTS = {
    RegNet_Y_400MF_Weights: regnet_y_400mf,
    RegNet_Y_400MF_Weights: regnet_y_400mf,
    RegNet_Y_800MF_Weights: regnet_y_800mf,
    RegNet_Y_1_6GF_Weights: regnet_y_1_6gf,
    RegNet_Y_3_2GF_Weights: regnet_y_3_2gf,
    RegNet_Y_8GF_Weights: regnet_y_8gf,
    RegNet_Y_16GF_Weights: regnet_y_16gf,
    RegNet_Y_32GF_Weights: regnet_y_32gf,
    # RegNet_Y_128GF_Weights: regnet_y_128gf,
    RegNet_X_400MF_Weights: regnet_x_400mf,
    RegNet_X_800MF_Weights: regnet_x_800mf,
    RegNet_X_1_6GF_Weights: regnet_x_1_6gf,
    RegNet_X_3_2GF_Weights: regnet_x_3_2gf,
    RegNet_X_8GF_Weights: regnet_x_8gf,
    RegNet_X_16GF_Weights: regnet_x_16gf,
    RegNet_X_32GF_Weights: regnet_x_32gf,
}

import numpy as np

LOGITS = {
    "y_400mf": np.array([]),
    "y_800mf": np.array([]),
    "y_1_6gf": np.array([]),
    "y_3_2gf": np.array([]),
    "y_8gf": np.array([]),
    "y_16gf": np.array([]),
    "y_32gf": np.array([]),
    "y_128gf": np.array([]),
    "x_400mf": np.array([]),
    "x_800mf": np.array([]),
    "x_1_6gf": np.array([]),
    "x_3_2gf400": np.array([]),
    "x_8gf": np.array([]),
    "x_16gf": np.array([]),
    "x_32gf": np.array([]),
    "r152": np.array([]),
    # "r152": np.array([0.8051, 0.0473, 0.0094]),
}


for weight, model in VARIANTS.items():
    model = model(weights=weight).to("cuda")
    model.eval()

    torch_output = torch.softmax(model(torch_img.cuda()), dim=1)
    torch_classes = torch.argsort(torch_output[0], descending=True)[:3]
    torch_logits = torch.take(torch_output[0], torch_classes)

    with open("output_4.txt", "w") as file:
        for key in LOGITS.keys():
            file.write(f" {key} : np.array({torch_logits}\n")

# print("Indices of the top 3 classes are:", torch_classes)
# print("Logits of the top 3 classes are:", torch_logits)
# print("Categories of the top 3 classes are:", [categories[i] for i in torch_classes])
