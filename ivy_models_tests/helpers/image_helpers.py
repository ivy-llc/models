import ivy
import numpy as np
from PIL import Image
from torchvision import transforms


def load_image_in_np(path):
    img = Image.open(path)
    return np.asarray(img)


def resize_img(img, new_size):
    img = np.resize(img, new_size)
    return ivy.asarray(img)


def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img -= ivy.array(mean)
    img /= ivy.array(std)
    return img


def crop_center(img, new_x, new_y):
    y, x, _ = img.shape
    start_x = x // 2 - (new_x // 2)
    start_y = y // 2 - (new_y // 2)
    return img[start_y : start_y + new_y, start_x : start_x + new_x, :]


def load_and_preprocess_img(
    path, new_size, crop, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    img = Image.open(path)
    compose = transforms.Compose(
        [
            transforms.Resize(new_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    img = compose(img)
    img = img.unsqueeze(0).permute(dims=(0, 2, 3, 1))
    return img.numpy()
