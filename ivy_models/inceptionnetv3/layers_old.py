import ivy
from typing import Optional, Callable, List, Tuple


class Inception_BasicConv2d(ivy.Module):
    """
    Conv block used in the InceptionV3 architecture.

    Args::
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size ([list | tuple]): kernel_shape for the block.
        stride (Optional[ivy.Module]): Stride value for the block.

    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super().__init__()

    def _build(self, *args, **kwargs):
        self.conv = ivy.Conv2D(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, with_bias=False)
        self.bn = ivy.BatchNorm2D(self.out_channels, eps=0.001)

    def _forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = ivy
        
        
class Inception_InceptionAux(ivy.Module):
    def __init__(self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = Inception_BasicConv2d
        self.in_channels = in_channels
        self.num_classes = num_classes

    def _build(self, *args, **kwargs):
        self.conv0 = self.conv_block(self.in_channels, 128, kernel_size=[1,1])
        self.conv1 = self.conv_block(128, 768, kernel_size=[5,5])
        self.conv1.stddev = 0.01
        self.fc = ivy.Linear(768, self.num_classes)
        self.fc.stddev = 0.001 
        
    def _forward(self, x):
      def forward(self, x):
        # N x 3 x 299 x 299
        x = self.conv2d_1a(x)
        # N x 32 x 149 x 149
        x = self.conv2d_2a(x)
        # N x 32 x 147 x 147
        x = self.conv2d_2b(x)
        # N x 64 x 147 x 147
        x = ivy.max_pool2d(x, 3, 2, 'valid', data_format='NHWC')
        # N x 64 x 73 x 73
        x = self.conv2d_3b(x)
        # N x 80 x 73 x 73
        x = self.conv2d_4a(x)
        # N x 192 x 71 x 71
        x = ivy.max_pool2d(x, 3, 2, 'valid', data_format='NHWC')
        # N x 192 x 35 x 35
        x = self.mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.mixed_6e(x)
        # N x 768 x 17 x 17
        aux = self.aux_logits(x)
        # N x 768 x 17 x 17
        x = self.mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.mixed_7c(x)
        # N x 2048 x 8 x 8
        x = ivy.avg_pool2d(x, 8, 1, 'valid', data_format='NHWC')
        # N x 2048 x 1 x 1
        x = ivy.flatten(x, start_dim=1)
        # N x 2048
        x = self.dropout(x)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux


class Inception_InceptionE(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = Inception_BasicConv2d
        self.in_channels = in_channels

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

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = ivy.concat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3),], axis=3)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = ivy.concat([self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl),], axis=3)

        branch_pool = ivy.avg_pool2d(x, [3,3], (1,1), [(1,1),(1,1)])
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        
        return outputs
    
    
class Inception_InceptionD(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = Inception_BasicConv2d
        self.in_channels = in_channels

    def _build(self, *args, **kwargs):
        self.branch3x3_1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        self.branch3x3_2 = self.conv_block(192, 320, kernel_size=[3,3], stride=2)

        self.branch7x7x3_1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        self.branch7x7x3_2 = self.conv_block(192, 192, kernel_size=[1,7], padding=[[0,0], [3,3]])
        self.branch7x7x3_3 = self.conv_block(192, 192, kernel_size=[7,1], padding=[[3,3], [0,0]])
        self.branch7x7x3_4 = self.conv_block(192, 192, kernel_size=[3,3], stride=2)

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = ivy.max_pool2d(x, [3,3], 2, 0)

        outputs = [branch3x3, branch7x7x3, branch_pool]
        outputs = ivy.concat(outputs, axis=3)

        return outputs
    
    
class Inception_InceptionC(ivy.Module):
    def __init__(self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = Inception_BasicConv2d
        self.in_channels = in_channels
        self.channels_7x7 = channels_7x7

    def _build(self, *args, **kwargs):
        self.branch1x1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])

        c7 = self.channels_7x7
        self.branch7x7_1 = self.conv_block(self.in_channels, c7, kernel_size=[1,1])
        self.branch7x7_2 = self.conv_block(c7, c7, kernel_size=[1, 7], padding=[[0,0],[3,3]])
        self.branch7x7_3 = self.conv_block(c7, 192, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        self.branch7x7dbl_1 = self.conv_block(self.in_channels, c7, kernel_size=[1,1])
        self.branch7x7dbl_2 = self.conv_block(c7, c7, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        self.branch7x7dbl_3 = self.conv_block(c7, c7, kernel_size=[1, 7], padding=[[0,0],[3,3]])
        self.branch7x7dbl_4 = self.conv_block(c7, c7, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        self.branch7x7dbl_5 = self.conv_block(c7, 192, kernel_size=[1, 7], padding=[[0,0],[3,3]])

        self.branch_pool = self.conv_block(self.in_channels, 192, kernel_size=[1,1])

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = ivy.avg_pool2d(x, [3,3], [1,1], [[1,1],[1,1]])
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)

        return outputs
    
class Inception_InceptionB(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = Inception_BasicConv2d
        self.in_channels = in_channels

    def _build(self, *args, **kwargs):
        self.branch3x3 = self.conv_block(self.in_channels, 384, kernel_size=[3,3], stride=2)

        self.branch3x3dbl_1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        self.branch3x3dbl_2 = self.conv_block(64, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        self.branch3x3dbl_3 = self.conv_block(96, 96, kernel_size=[3,3], stride=2)

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        branch3x3 = self.branch3x3(x)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = ivy.max_pool2d(x, [3,3], 2, 0)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)

        return outputs
    
    
class Inception_InceptionA(ivy.Module):
    def __init__(self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = Inception_BasicConv2d
        self.in_channels = in_channels
        self.pool_features = pool_features

    def _build(self, *args, **kwargs):
        self.branch1x1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])

        self.branch5x5_1 = self.conv_block(self.in_channels, 48, kernel_size=[1,1])
        self.branch5x5_2 = self.conv_block(48, 64, kernel_size=[5,5], padding=[[2,2],[2,2]])

        self.branch3x3dbl_1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        self.branch3x3dbl_2 = self.conv_block(64, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        self.branch3x3dbl_3 = self.conv_block(96, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])

        self.branch_pool = self.conv_block(self.in_channels, self.pool_features, kernel_size=[1,1])


    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = ivy.avg_pool2d(x, [3,3], [1,1], [[1,1],[1,1]])
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)

        return outputs
    