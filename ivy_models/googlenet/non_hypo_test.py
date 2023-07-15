import os
import ivy
import pytest
import numpy as np
from googlenet import inceptionNet_v1
from ivy_models_tests import helpers

import torch
from torchvision.models import googlenet, get_model_weights
import torchvision.transforms as transforms
from PIL import Image


def test():
    model = inceptionNet_v1(pretrained=False)
    output = model(torch.randn(1, 3, 224, 224))
    print(output.size())

test()