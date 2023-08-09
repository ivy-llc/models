import ivy
import numpy as np
import random
from PIL import Image, ImageFilter, ImageOps
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
    path,
    new_size,
    crop,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    data_format="NHWC",
    to_ivy=False,
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
    img = img.unsqueeze(0)
    if data_format == "NHWC":
        img = img.permute((0, 2, 3, 1))
    return ivy.array(img.numpy()) if to_ivy else img.numpy()


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

