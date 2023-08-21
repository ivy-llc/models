from typing import Callable, Optional, Union, Tuple, List
import ivy

import sys
sys.path.append("/ivy_models/log_sys/pf.py")
from log_sys.pf import *


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
        pf(f"BasicConv2d | build | done 0/2")
        self.conv = ivy.Conv2D(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, with_bias=False)
        pf(f"BasicConv2d | build | done 2/2")
        self.bn = ivy.BatchNorm2D(self.out_channels, eps=0.001)
        pf(f"BasicConv2d | build | done 2/2")

    def _forward(self, x):
        """Forward pass method for the module."""
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


class InceptionAux(ivy.Module):
    def __init__(self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        self.num_classes = num_classes
        super().__init__()

    def _build(self, *args, **kwargs):
        self.conv0 = self.conv_block(self.in_channels, 128, kernel_size=[1,1])
        self.conv1 = self.conv_block(128, 768, kernel_size=[5,5])
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = ivy.Linear(768, self.num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def _forward(self, x):
        # N x 768 x 17 x 17
        pf(f"InceptionAux | input shape is:{x.shape}")
        x = ivy.avg_pool2d(x, [5,5], 3, 'valid', data_format='NHWC')
        pf(f"InceptionAux | done 1/8, output shape is:{x.shape}")

        # N x 768 x 5 x 5
        x = self.conv0(x)
        pf("InceptionAux | done 2/8")

        # N x 128 x 5 x 5
        x = self.conv1(x)
        pf("InceptionAux | done 3/8")

        # N x 768 x 1 x 1
        # Adaptive average pooling
        pf(f"InceptionAux | input shape to adaptive_avg_pool2d is:{x.shape}")
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        pf(f"InceptionAux | permuted input shape to adaptive_avg_pool2d is:{x.shape}")
        x = ivy.adaptive_avg_pool2d(x, (1, 1))
        pf(f"InceptionAux | output shape from adaptive_avg_pool2d is:{x.shape}")
        x = ivy.permute_dims(x, (0, 2, 3, 1))
        pf(f"InceptionAux | permuted output shape from adaptive_avg_pool2d is:{x.shape}")
        pf("InceptionAux | done 4/8")

        # N x 768 x 1 x 1
        x = ivy.flatten(x, start_dim=1)
        pf("InceptionAux | done 5/8")

        # N x 768
        x = self.fc(x)
        pf("InceptionAux | done 8/8")
        # N x 1000
        return x


def test_InceptionAux():
    random_test_tensor = ivy.random_normal(shape=(1, 17, 17, 768))
    pf(f"InceptionAux | random_test_tensor shape is: {random_test_tensor.shape}")

    block = InceptionAux(768, 1000)
    block(random_test_tensor)
    pf("InceptionAux | Test Successfull!")

test_InceptionAux()


class InceptionA(ivy.Module):
    def __init__(self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        self.pool_features = pool_features
        super().__init__()

    def _build(self, *args, **kwargs):
        self.branch1x1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        pf(f"layer 1/7 built")

        self.branch5x5_1 = self.conv_block(self.in_channels, 48, kernel_size=[1,1])
        pf(f"layer 2/7 built")
        self.branch5x5_2 = self.conv_block(48, 64, kernel_size=[5,5], padding=[[2,2],[2,2]])
        pf(f"layer 3/7 built")

        self.branch3x3dbl_1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        pf(f"layer 4/7 built")
        self.branch3x3dbl_2 = self.conv_block(64, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        pf(f"layer 5/7 built")
        self.branch3x3dbl_3 = self.conv_block(96, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        pf(f"layer 6/7 built")

        self.branch_pool = self.conv_block(self.in_channels, self.pool_features, kernel_size=[1,1])
        pf(f"layer 7/7 built")

#         self.avg_pool = ivy.AvgPool2D(3,1,1)

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        pf(f"input shape is:{x.shape}")

        branch1x1 = self.branch1x1(x)
        pf(f"InceptionA | branch1x1 1/20, output shape is: {branch1x1.shape}")

        branch5x5 = self.branch5x5_1(x)
        pf(f"InceptionA | one 2/20, output shape is: {branch5x5.shape}")
        branch5x5 = self.branch5x5_2(branch5x5)
        pf(f"InceptionA | branch5x5_1 3/20, output shape is: {branch5x5.shape}")

        branch3x3dbl = self.branch3x3dbl_1(x)
        pf(f"InceptionA | one 4/20, output shape is: {branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        pf(f"InceptionA | one 5/20, output shape is: {branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        pf(f"InceptionA | branch3x3dbl_1 6/20, output shape is: {branch3x3dbl.shape}")

        branch_pool = ivy.avg_pool2d(x, [3,3], [1,1], [[1,1],[1,1]])
#         branch_pool = self.avg_pool(x)
        pf(f"InceptionA | one 7/20, output shape is: {branch_pool.shape}")
        branch_pool = self.branch_pool(branch_pool)
        pf(f"InceptionA | branch_pool 8/20, output shape is: {branch_pool.shape}")

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        pf(f"InceptionA | outputs 20/20")

        return outputs


def test_InceptionA():
    random_test_tensor = ivy.random_normal(shape=(1, 35, 35, 192))
    pf(f"random_test_tensor shape is: {random_test_tensor.shape}")

    # N x 192 x 35 x 35
    block = InceptionA(192, pool_features=32)
    block(random_test_tensor)
    # N x 256 x 35 x 35
    pf("Test Successfull!")

test_InceptionA()


# note:
# kernal_size in [1,3] list foramt
# and padding in list[list, list] foramt ex: [[1,1],[3,3]] if both dims are unequal, else just single ex: list [3,3]
class InceptionB(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        super().__init__()

    def _build(self, *args, **kwargs):
        self.branch3x3 = self.conv_block(self.in_channels, 384, kernel_size=[3,3], stride=2)
        pf(f"layer 1/4 built")

        self.branch3x3dbl_1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        pf(f"layer 2/4 built")
        self.branch3x3dbl_2 = self.conv_block(64, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        pf(f"layer 3/4 built")
        self.branch3x3dbl_3 = self.conv_block(96, 96, kernel_size=[3,3], stride=2)
        pf(f"layer 4/4 built")

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        pf(f"input shape is:{x.shape}")

        branch3x3 = self.branch3x3(x)
        pf(f"one 1/20, output shape is:{branch3x3.shape}")

        branch3x3dbl = self.branch3x3dbl_1(x)
        pf(f"one 2/20, output shape is:{branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        pf(f"one 3/20, output shape is:{branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        pf(f"one 4/20, output shape is:{branch3x3dbl.shape}")

        branch_pool = ivy.max_pool2d(x, [3,3], 2, 0)
        pf(f"one 20/20, output shape is:{branch_pool.shape}")

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        pf(f"one 20/20")

        return outputs


def test_InceptionB():
    random_test_tensor = ivy.random_normal(shape=(1, 35, 35, 288))
    pf(f"random_test_tensor shape is: {random_test_tensor.shape}")

    # N x 288 x 35 x 35
    block = InceptionB(288)
    block(random_test_tensor)
    # N x 768 x 17 x 17
    pf("Test Successfull!")

test_InceptionB()



# note:
# kernal_size in [1,3] list foramt
# and padding in list[list, list] foramt ex: [[1,1],[3,3]] if both dims are unequal, else just single ex: list [3,3]
class InceptionC(ivy.Module):
    def __init__(self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        self.channels_7x7 = channels_7x7
        super().__init__()

    def _build(self, *args, **kwargs):
        self.branch1x1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        pf(f"layer 1/10 built")

        c7 = self.channels_7x7
        self.branch7x7_1 = self.conv_block(self.in_channels, c7, kernel_size=[1,1])
        pf(f"layer 2/10 built")
        self.branch7x7_2 = self.conv_block(c7, c7, kernel_size=[1, 7], padding=[[0,0],[3,3]])
        pf(f"layer 3/10 built")
        self.branch7x7_3 = self.conv_block(c7, 192, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        pf(f"layer 4/10 built")
        self.branch7x7dbl_1 = self.conv_block(self.in_channels, c7, kernel_size=[1,1])
        pf(f"layer 5/10 built")
        self.branch7x7dbl_2 = self.conv_block(c7, c7, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        pf(f"layer 6/10 built")
        self.branch7x7dbl_3 = self.conv_block(c7, c7, kernel_size=[1, 7], padding=[[0,0],[3,3]])
        pf(f"layer 7/10 built")
        self.branch7x7dbl_4 = self.conv_block(c7, c7, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        pf(f"layer 8/10 built")
        self.branch7x7dbl_5 = self.conv_block(c7, 192, kernel_size=[1, 7], padding=[[0,0],[3,3]])
        pf(f"layer 9/10 built")

        self.branch_pool = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        pf(f"layer 10/10 built")

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        pf(f"input shape is:{x.shape}")

        branch1x1 = self.branch1x1(x)
        pf(f"one 1/20, output shape is:{branch1x1.shape}")

        branch7x7 = self.branch7x7_1(x)
        pf(f"one 2/20, output shape is:{branch7x7.shape}")
        branch7x7 = self.branch7x7_2(branch7x7)
        pf(f"one 3/20, output shape is:{branch7x7.shape}")
        branch7x7 = self.branch7x7_3(branch7x7)
        pf(f"one 4/20, output shape is:{branch7x7.shape}")

        branch7x7dbl = self.branch7x7dbl_1(x)
        pf(f"one 5/20, output shape is:{branch7x7dbl.shape}")
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        pf(f"one 6/20, output shape is:{branch7x7dbl.shape}")
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        pf(f"one 7/20, output shape is:{branch7x7dbl.shape}")
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        pf(f"one 8/20, output shape is:{branch7x7dbl.shape}")
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        pf(f"one 9/20, output shape is:{branch7x7dbl.shape}")

        branch_pool = ivy.avg_pool2d(x, [3,3], [1,1], [[1,1],[1,1]])
        pf(f"one 10/20, output shape is:{branch_pool.shape}")
        branch_pool = self.branch_pool(branch_pool)
        pf(f"one 11/20, output shape is:{branch_pool.shape}")

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        pf(f"one 20/20")

        return outputs


def test_InceptionC():
    random_test_tensor = ivy.random_normal(shape=(1, 17, 17, 768))
    pf(f"random_test_tensor shape is: {random_test_tensor.shape}")

    # N x 768 x 17 x 17
    block = InceptionC(768, channels_7x7=128)
    block(random_test_tensor)
    # N x 768 x 17 x 17
    pf("Test Successfull!")

test_InceptionC()



# note:
# kernal_size in [1,3] list foramt
# and padding in list[list, list] foramt ex: [[1,1],[3,3]] if both dims are unequal, else just single ex: list [3,3]
class InceptionD(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        super().__init__()

    def _build(self, *args, **kwargs):
        self.branch3x3_1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        pf(f"layer 1/6 built")
        self.branch3x3_2 = self.conv_block(192, 320, kernel_size=[3,3], stride=2)
        pf(f"layer 2/6 built")

        self.branch7x7x3_1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        pf(f"layer 3/6 built")
        self.branch7x7x3_2 = self.conv_block(192, 192, kernel_size=[1,7], padding=[[0,0], [3,3]])
        pf(f"layer 4/6 built")
        self.branch7x7x3_3 = self.conv_block(192, 192, kernel_size=[7,1], padding=[[3,3], [0,0]])
        pf(f"layer 5/6 built")
        self.branch7x7x3_4 = self.conv_block(192, 192, kernel_size=[3,3], stride=2)
        pf(f"layer 6/6 built")

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        pf(f"input shape is:{x.shape}")

        branch3x3 = self.branch3x3_1(x)
        pf(f"one 1/20, output shape is:{branch3x3.shape}")
        branch3x3 = self.branch3x3_2(branch3x3)
        pf(f"one 2/20, output shape is:{branch3x3.shape}")

        branch7x7x3 = self.branch7x7x3_1(x)
        pf(f"one 3/20, output shape is:{branch7x7x3.shape}")
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        pf(f"one 4/20, output shape is:{branch7x7x3.shape}")
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        pf(f"one 5/20, output shape is:{branch7x7x3.shape}")
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        pf(f"one 6/20, output shape is:{branch7x7x3.shape}")

        branch_pool = ivy.max_pool2d(x, [3,3], 2, 0)
        pf(f"one 7/20, output shape is:{branch_pool.shape}")

        outputs = [branch3x3, branch7x7x3, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        pf(f"one 20/20")

        return outputs


def test_InceptionD():
    random_test_tensor = ivy.random_normal(shape=(1, 17, 17, 768))
    pf(f"random_test_tensor shape is: {random_test_tensor.shape}")

    # N x 768 x 17 x 17
    block = InceptionD(768)
    block(random_test_tensor)
    # N x 1280 x 8 x 8
    pf("Test Successfull!")

test_InceptionD()




class InceptionE(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        super().__init__()

    def _build(self, *args, **kwargs):
        self.branch1x1 = self.conv_block(self.in_channels, 320, kernel_size=[1,1])

        self.branch3x3_1 = self.conv_block(self.in_channels, 384, kernel_size=[1,1])
        self.branch3x3_2a = self.conv_block(384, 384, kernel_size=[1, 3], padding=[[0,0], [1,1]])
        self.branch3x3_2b = self.conv_block(384, 384, kernel_size=[3, 1], padding=[[1,1], [0,0]])

        self.branch3x3dbl_1 = self.conv_block(self.in_channels, 448, kernel_size=[1,1])
        self.branch3x3dbl_2 = self.conv_block(448, 384, kernel_size=[3,3], padding=1)
        self.branch3x3dbl_3a = self.conv_block(384, 384, kernel_size=[1, 3], padding=[[0,0], [1,1]])
        self.branch3x3dbl_3b = self.conv_block(384, 384, kernel_size=[3, 1], padding=[[1,1], [0,0]])

        self.branch_pool = self.conv_block(self.in_channels, 192, kernel_size=[1,1])

#         self.avg_pool = ivy.AvgPool2D([3,3], (1,1), [[1,1],[1,1]])

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        pf(f"input shape is:{x.shape}")

        branch1x1 = self.branch1x1(x)
        pf(f"1/20, branch1x1 output shape is:{branch1x1.shape}")

        branch3x3 = self.branch3x3_1(x)
        pf(f"2/20, branch3x3 output shape is:{branch3x3.shape}")
        branch3x3 = ivy.concat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3),], axis=3)
        pf(f"3/20, branch3x3 output shape is:{branch3x3.shape}")

        branch3x3dbl = self.branch3x3dbl_1(x)
        pf(f"4/20, branch3x3dbl output shape is:{branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        pf(f"5/20, branch3x3dbl output shape is:{branch3x3dbl.shape}")
        branch3x3dbl = ivy.concat([self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl),], axis=3)
        pf(f"6/20, branch3x3dbl output shape is:{branch3x3dbl.shape}")


        branch_pool = ivy.avg_pool2d(x, [3,3], (1,1), [(1,1),(1,1)])
#         branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        pf(f"7/20, branch_pool output shape is:{branch_pool.shape}")

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        pf(f"20/20")
        return outputs


def test_InceptionE():
    random_test_tensor = ivy.random_normal(shape=(1, 8, 8, 1280))
    pf(f"random_test_tensor shape is: {random_test_tensor.shape}")

    # N x 1280 x 8 x 8
    block = InceptionE(1280)
    block(random_test_tensor)
    # N x 2048 x 8 x 8
    pf("Test Successfull!")

test_InceptionE()




