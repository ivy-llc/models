import ivy
from typing import *

class BasicConv2d(ivy.Module):
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
        
        
class InceptionAux(ivy.Module):
    def __init__(self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        self.num_classes = num_classes

    def _build(self, *args, **kwargs):
        self.conv0 = self.conv_block(self.in_channels, 128, kernel_size=[1,1])
        self.conv1 = self.conv_block(128, 768, kernel_size=[5,5])
        self.conv1.stddev = 0.01
        self.fc = ivy.Linear(768, self.num_classes)
        self.fc.stddev = 0.001 
        
    def _forward(self, x):
        # N x 768 x 17 x 17
        x = ivy.avg_pool2d(x, [5,5], 3, 'valid', data_format='NHWC')

        # N x 768 x 5 x 5
        x = self.conv0(x)
        #display("InceptionAux | done 2/8")

        # N x 128 x 5 x 5
        x = self.conv1(x)
        #display("InceptionAux | done 3/8")

        # N x 768 x 1 x 1
        # Adaptive average pooling
        #display(f"InceptionAux | input shape to adaptive_avg_pool2d is:{x.shape}")
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        #display(f"InceptionAux | permuted input shape to adaptive_avg_pool2d is:{x.shape}")
        x = ivy.adaptive_avg_pool2d(x, (1, 1))
        #display(f"InceptionAux | output shape from adaptive_avg_pool2d is:{x.shape}")
        x = ivy.permute_dims(x, (0, 2, 3, 1))
        #display(f"InceptionAux | permuted output shape from adaptive_avg_pool2d is:{x.shape}")
        #display("InceptionAux | done 4/8")

        # N x 768 x 1 x 1
        x = ivy.flatten(x, start_dim=1)
        #display("InceptionAux | done 5/8")

        # N x 768
        x = self.fc(x)
        #display("InceptionAux | done 8/8")
        # N x 1000
        return x


class InceptionE(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = BasicConv2d
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

#         self.avg_pool = ivy.AvgPool2D([3,3], (1,1), [[1,1],[1,1]])

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        #display(f"input shape is:{x.shape}")

        branch1x1 = self.branch1x1(x)
        #display(f"1/20, branch1x1 output shape is:{branch1x1.shape}")

        branch3x3 = self.branch3x3_1(x)
        #display(f"2/20, branch3x3 output shape is:{branch3x3.shape}")
        branch3x3 = ivy.concat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3),], axis=3)
        #display(f"3/20, branch3x3 output shape is:{branch3x3.shape}")

        branch3x3dbl = self.branch3x3dbl_1(x)
        #display(f"4/20, branch3x3dbl output shape is:{branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #display(f"5/20, branch3x3dbl output shape is:{branch3x3dbl.shape}")
        branch3x3dbl = ivy.concat([self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl),], axis=3)
        #display(f"6/20, branch3x3dbl output shape is:{branch3x3dbl.shape}")

        #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        import pprint
        # pprint.pprint(x)

        #--------------------------------
        import json
        import numpy as np

        # Create a NumPy array
        arr = np.array(x)

        # Convert the NumPy array to a Python list
        arr_list = arr.tolist()

        # Define the path and filename for the JSON file
        file_path = '/content/file.json'

        # Save the NumPy array as JSON
        with open(file_path, 'w') as json_file:
            json.dump(arr_list, json_file)
        #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


        branch_pool = ivy.avg_pool2d(x, [3,3], (1,1), [(1,1),(1,1)])
#         branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        #display(f"7/20, branch_pool output shape is:{branch_pool.shape}")

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        #display(f"20/20")
        return outputs
    
    
class InceptionD(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels

    def _build(self, *args, **kwargs):
        self.branch3x3_1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        #display(f"layer 1/6 built")
        self.branch3x3_2 = self.conv_block(192, 320, kernel_size=[3,3], stride=2)
        #display(f"layer 2/6 built")

        self.branch7x7x3_1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        #display(f"layer 3/6 built")
        self.branch7x7x3_2 = self.conv_block(192, 192, kernel_size=[1,7], padding=[[0,0], [3,3]])
        #display(f"layer 4/6 built")
        self.branch7x7x3_3 = self.conv_block(192, 192, kernel_size=[7,1], padding=[[3,3], [0,0]])
        #display(f"layer 5/6 built")
        self.branch7x7x3_4 = self.conv_block(192, 192, kernel_size=[3,3], stride=2)
        #display(f"layer 6/6 built")

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        #display(f"input shape is:{x.shape}")

        branch3x3 = self.branch3x3_1(x)
        #display(f"one 1/20, output shape is:{branch3x3.shape}")
        branch3x3 = self.branch3x3_2(branch3x3)
        #display(f"one 2/20, output shape is:{branch3x3.shape}")

        branch7x7x3 = self.branch7x7x3_1(x)
        #display(f"one 3/20, output shape is:{branch7x7x3.shape}")
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        #display(f"one 4/20, output shape is:{branch7x7x3.shape}")
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        #display(f"one 5/20, output shape is:{branch7x7x3.shape}")
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        #display(f"one 6/20, output shape is:{branch7x7x3.shape}")

        branch_pool = ivy.max_pool2d(x, [3,3], 2, 0)
        #display(f"one 7/20, output shape is:{branch_pool.shape}")

        outputs = [branch3x3, branch7x7x3, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        #display(f"one 20/20")

        return outputs
    
    
class InceptionC(ivy.Module):
    def __init__(self, in_channels: int, channels_7x7: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        self.channels_7x7 = channels_7x7

    def _build(self, *args, **kwargs):
        self.branch1x1 = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        #display(f"layer 1/10 built")

        c7 = self.channels_7x7
        self.branch7x7_1 = self.conv_block(self.in_channels, c7, kernel_size=[1,1])
        #display(f"layer 2/10 built")
        self.branch7x7_2 = self.conv_block(c7, c7, kernel_size=[1, 7], padding=[[0,0],[3,3]])
        #display(f"layer 3/10 built")
        self.branch7x7_3 = self.conv_block(c7, 192, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        #display(f"layer 4/10 built")
        self.branch7x7dbl_1 = self.conv_block(self.in_channels, c7, kernel_size=[1,1])
        #display(f"layer 5/10 built")
        self.branch7x7dbl_2 = self.conv_block(c7, c7, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        #display(f"layer 6/10 built")
        self.branch7x7dbl_3 = self.conv_block(c7, c7, kernel_size=[1, 7], padding=[[0,0],[3,3]])
        #display(f"layer 7/10 built")
        self.branch7x7dbl_4 = self.conv_block(c7, c7, kernel_size=[7, 1], padding=[[3,3],[0,0]])
        #display(f"layer 8/10 built")
        self.branch7x7dbl_5 = self.conv_block(c7, 192, kernel_size=[1, 7], padding=[[0,0],[3,3]])
        #display(f"layer 9/10 built")

        self.branch_pool = self.conv_block(self.in_channels, 192, kernel_size=[1,1])
        #display(f"layer 10/10 built")

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        #display(f"input shape is:{x.shape}")

        branch1x1 = self.branch1x1(x)
        #display(f"one 1/20, output shape is:{branch1x1.shape}")

        branch7x7 = self.branch7x7_1(x)
        #display(f"one 2/20, output shape is:{branch7x7.shape}")
        branch7x7 = self.branch7x7_2(branch7x7)
        #display(f"one 3/20, output shape is:{branch7x7.shape}")
        branch7x7 = self.branch7x7_3(branch7x7)
        #display(f"one 4/20, output shape is:{branch7x7.shape}")

        branch7x7dbl = self.branch7x7dbl_1(x)
        #display(f"one 5/20, output shape is:{branch7x7dbl.shape}")
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        #display(f"one 6/20, output shape is:{branch7x7dbl.shape}")
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        #display(f"one 7/20, output shape is:{branch7x7dbl.shape}")
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        #display(f"one 8/20, output shape is:{branch7x7dbl.shape}")
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        #display(f"one 9/20, output shape is:{branch7x7dbl.shape}")

        branch_pool = ivy.avg_pool2d(x, [3,3], [1,1], [[1,1],[1,1]])
        #display(f"one 10/20, output shape is:{branch_pool.shape}")
        branch_pool = self.branch_pool(branch_pool)
        #display(f"one 11/20, output shape is:{branch_pool.shape}")

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        #display(f"one 20/20")

        return outputs
    
    
class InceptionB(ivy.Module):
    def __init__(self, in_channels: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels

    def _build(self, *args, **kwargs):
        self.branch3x3 = self.conv_block(self.in_channels, 384, kernel_size=[3,3], stride=2)
        #display(f"layer 1/4 built")

        self.branch3x3dbl_1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        #display(f"layer 2/4 built")
        self.branch3x3dbl_2 = self.conv_block(64, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        #display(f"layer 3/4 built")
        self.branch3x3dbl_3 = self.conv_block(96, 96, kernel_size=[3,3], stride=2)
        #display(f"layer 4/4 built")

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        #display(f"input shape is:{x.shape}")

        branch3x3 = self.branch3x3(x)
        #display(f"one 1/20, output shape is:{branch3x3.shape}")

        branch3x3dbl = self.branch3x3dbl_1(x)
        #display(f"one 2/20, output shape is:{branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #display(f"one 3/20, output shape is:{branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        #display(f"one 4/20, output shape is:{branch3x3dbl.shape}")

        branch_pool = ivy.max_pool2d(x, [3,3], 2, 0)
        #display(f"one 20/20, output shape is:{branch_pool.shape}")

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        #display(f"one 20/20")

        return outputs
    
    
    
class InceptionA(ivy.Module):
    def __init__(self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        self.pool_features = pool_features

    def _build(self, *args, **kwargs):
        self.branch1x1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        #display(f"layer 1/7 built")

        self.branch5x5_1 = self.conv_block(self.in_channels, 48, kernel_size=[1,1])
        #display(f"layer 2/7 built")
        self.branch5x5_2 = self.conv_block(48, 64, kernel_size=[5,5], padding=[[2,2],[2,2]])
        #display(f"layer 3/7 built")

        self.branch3x3dbl_1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        #display(f"layer 4/7 built")
        self.branch3x3dbl_2 = self.conv_block(64, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        #display(f"layer 5/7 built")
        self.branch3x3dbl_3 = self.conv_block(96, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        #display(f"layer 6/7 built")

        self.branch_pool = self.conv_block(self.in_channels, self.pool_features, kernel_size=[1,1])
        #display(f"layer 7/7 built")

#         self.avg_pool = ivy.AvgPool2D(3, 1, 1)

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        #display(f"input shape is:{x.shape}")

        branch1x1 = self.branch1x1(x)
        #display(f"one 1/20")

        branch5x5 = self.branch5x5_1(x)
        #display(f"one 2/20")
        branch5x5 = self.branch5x5_2(branch5x5)
        #display(f"one 3/20")

        branch3x3dbl = self.branch3x3dbl_1(x)
        #display(f"one 4/20")
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #display(f"one 5/20")
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        #display(f"one 6/20")

        branch_pool = ivy.avg_pool2d(x, [3,3], [1,1], [[1,1],[1,1]])
#         branch_pool = self.avg_pool(x)
        #display(f"one 7/20")
        branch_pool = self.branch_pool(branch_pool)
        #display(f"one 8/20")

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        #display(f"one 20/20")

        return outputs
    
    
class InceptionA(ivy.Module):
    def __init__(self, in_channels: int, pool_features: int, conv_block: Optional[Callable[..., ivy.Module]] = None) -> None:
        super().__init__()
        if conv_block is None:
            self.conv_block = BasicConv2d
        self.in_channels = in_channels
        self.pool_features = pool_features

    def _build(self, *args, **kwargs):
        self.branch1x1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        #display(f"layer 1/7 built")

        self.branch5x5_1 = self.conv_block(self.in_channels, 48, kernel_size=[1,1])
        #display(f"layer 2/7 built")
        self.branch5x5_2 = self.conv_block(48, 64, kernel_size=[5,5], padding=[[2,2],[2,2]])
        #display(f"layer 3/7 built")

        self.branch3x3dbl_1 = self.conv_block(self.in_channels, 64, kernel_size=[1,1])
        #display(f"layer 4/7 built")
        self.branch3x3dbl_2 = self.conv_block(64, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        #display(f"layer 5/7 built")
        self.branch3x3dbl_3 = self.conv_block(96, 96, kernel_size=[3,3], padding=[[1,1],[1,1]])
        #display(f"layer 6/7 built")

        self.branch_pool = self.conv_block(self.in_channels, self.pool_features, kernel_size=[1,1])
        #display(f"layer 7/7 built")

#         self.avg_pool = ivy.AvgPool2D(3,1,1)

    def _forward(self, x: ivy.Array) -> List[ivy.Array]:
        #display(f"input shape is:{x.shape}")

        branch1x1 = self.branch1x1(x)
        #display(f"InceptionA | branch1x1 1/20, output shape is: {branch1x1.shape}")

        branch5x5 = self.branch5x5_1(x)
        #display(f"InceptionA | one 2/20, output shape is: {branch5x5.shape}")
        branch5x5 = self.branch5x5_2(branch5x5)
        #display(f"InceptionA | branch5x5_1 3/20, output shape is: {branch5x5.shape}")

        branch3x3dbl = self.branch3x3dbl_1(x)
        #display(f"InceptionA | one 4/20, output shape is: {branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #display(f"InceptionA | one 5/20, output shape is: {branch3x3dbl.shape}")
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        #display(f"InceptionA | branch3x3dbl_1 6/20, output shape is: {branch3x3dbl.shape}")

        branch_pool = ivy.avg_pool2d(x, [3,3], [1,1], [[1,1],[1,1]])
#         branch_pool = self.avg_pool(x)
        #display(f"InceptionA | one 7/20, output shape is: {branch_pool.shape}")
        branch_pool = self.branch_pool(branch_pool)
        #display(f"InceptionA | branch_pool 8/20, output shape is: {branch_pool.shape}")

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = ivy.concat(outputs, axis=3)
        #display(f"InceptionA | outputs 20/20")

        return outputs
    
    