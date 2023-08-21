from typing import Callable, Optional, Union, Tuple, List
import ivy


class BasicConv2d(ivy.Module):
    """
    Basic block used in the ResNet architecture.

    Args::
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride value for the block. Defaults to 1.
        kernel_size (int): size of kernel.
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[List, Tuple],
        stride: int = 1, 
        padding: int = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super(BasicConv2d, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv = ivy.Conv2D(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, with_bias=False)
        self.bn = ivy.BatchNorm2D(self.out_channels, eps=0.001)

    def _forward(self, x):
        """Forward pass method for the module."""
        x = self.conv(x)
        x = self.bn(x)
        x = ivy.relu(x)
        return x
    