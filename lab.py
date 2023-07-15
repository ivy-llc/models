
import ivy
ivy.set_backend("torch")
import numpy as np
from ivy_models_tests import helpers

import torch
from torchvision.models import googlenet, get_model_weights
import torchvision.transforms as transforms
from PIL import Image


# # Load and preprocess the cat image
# image_path = '/models/images/cat.jpg'
# image = Image.open(image_path).convert('RGB')
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # 3-3 values are calculated mean and std on imagenet dataset for 3 channels each
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# input_tensor = transform(image).unsqueeze(0)


# Load image
img = helpers.load_and_preprocess_img(
    "/models/images/cat.jpg", 256, 224
)
print(ivy.shape(img))
# print(ivy.conv2d(img, [1,1], 1, 1))