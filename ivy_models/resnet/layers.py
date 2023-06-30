from typing import Optional
import ivy


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    dilations: int = 1,
) -> ivy.Conv2D:
    """3x3 convolution with padding"""
    return ivy.Conv2D(
        in_planes,
        out_planes,
        [3, 3],
        stride,
        dilations,
        with_bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> ivy.Conv2D:
    """1x1 convolution"""
    return ivy.Conv2D(in_planes, out_planes, [1, 1], stride, 0, with_bias=False)


class BasicBlock(ivy.Module):
    """
    Basic block used in the ResNet architecture.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride value for the block. Defaults to 1.
        downsample (Optional[ivy.Module]): Downsample module for the block. Defaults to None.
        base_width (int): The base width of the block. Defaults to 64.
        dilation (int): Dilation rate of the block. Defaults to 1.

    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[ivy.Module] = None,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        self.norm_layer = ivy.BatchNorm2D
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if base_width != 64:
            raise ValueError("BasicBlock only supports base_width=64")

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        super(BasicBlock, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv1 = conv3x3(self.inplanes, self.planes, self.stride)
        self.bn1 = self.norm_layer(self.planes)
        self.relu = ivy.ReLU()
        self.conv2 = conv3x3(self.planes, self.planes)
        self.bn2 = self.norm_layer(self.planes)
        self.downsample = self.downsample
        self.stride = self.stride

    def _forward(self, x):
        """Forward pass method for the module."""

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(ivy.Module):
    """
    Bottleneck block used in the ResNet architecture.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int): Stride value for the block. Defaults to 1.
        downsample (Optional[ivy.Module]): Downsample module for the block. Defaults to None.
        base_width (int): The base width of the block. Defaults to 64.
        dilation (int): Dilation rate of the block. Defaults to 1.

    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[ivy.Module] = None,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        self.norm_layer = ivy.BatchNorm2D
        self.width = int(planes * (base_width / 64.0))
        self.inplanes = inplanes
        self.planes = planes
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        super(Bottleneck, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv1 = conv1x1(self.inplanes, self.width)
        self.bn1 = self.norm_layer(self.width)
        self.conv2 = conv3x3(self.width, self.width, self.stride, self.dilation)
        self.bn2 = self.norm_layer(self.width)
        self.conv3 = conv1x1(self.width, self.planes * self.expansion)
        self.bn3 = self.norm_layer(self.planes * self.expansion)
        self.relu = ivy.ReLU()
        self.downsample = self.downsample

    def _forward(self, x):
        """Forward pass method for the module."""

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
