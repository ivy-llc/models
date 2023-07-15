import os
import ivy
import pytest
import numpy as np
from my_resnet import ResNetModel
from ivy_models_tests import helpers

import torch
from torchvision.models import googlenet, get_model_weights
import torchvision.transforms as transforms
from PIL import Image


def test():
    model = ResNetModel(name='resnet50', head='mlp', feat_dim=1000)
    output = model(torch.randn(1, 3, 224, 224))
    print(output.size())

test()