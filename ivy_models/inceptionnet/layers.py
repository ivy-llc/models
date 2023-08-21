from typing import Callable, Optional, Union, Tuple
import ivy

import sys
sys.path.append("/ivy_models/log_sys/pf.py")
from log_sys.pf import *

class BasicConv2d(ivy.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super().__init__()

    def _build(self, *args, **kwargs):
        pf(f"BasicConv2d | build | done 1/2")
        self.conv = ivy.Conv2D(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, with_bias=False)
        pf(f"BasicConv2d | build | done 2/2")
        self.bn = ivy.BatchNorm2D(self.out_channels, eps=0.001)

    def _forward(self, x):
        pf(f"BasicConv2d | forward | input shape is: {ivy.shape(x)} | done 0/3")
        x = self.conv(x)
        pf(f"BasicConv2d | forward | shape is: {ivy.shape(x)} | done 1/3")
        x = self.bn(x)
        pf(f"BasicConv2d | forward | shape is: {ivy.shape(x)} | done 2/3")
        x = ivy.relu(x)
        pf(f"BasicConv2d | forward | output shape is: {ivy.shape(x)} | done 3/3")
        return x

def test_BasicConv2d():
    # N x 768 x 5 x 5
    random_test_tensor = ivy.random_normal(shape=(1, 5, 5, 768))
    pf(f"BasicConv2d | Test | input shape is: {random_test_tensor.shape}")

    block = BasicConv2d(768, 128, [1,1])
    block(random_test_tensor)
    # N x 128 x 5 x 5
    pf(f"BasicConv2d | Test | Test Successfull!")
    pf(f"||")

test_BasicConv2d()