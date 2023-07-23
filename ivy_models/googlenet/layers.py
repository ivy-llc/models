from typing import Optional
import ivy

class ConvBlock(ivy.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super(ConvBlock, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv = ivy.Conv2D(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, with_bias=False)
        self.bn = ivy.BatchNorm2D(self.out_channels, eps=0.001)
        self.activation = ivy.ReLU()

    def _forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class Inception(ivy.Module):
    def __init__(
        self,
        in_channels,
        num1x1,
        num3x3_reduce,
        num3x3,
        num5x5_reduce,
        num5x5,
        pool_proj,
    ):
        self.in_channels = in_channels
        self.num1x1 = num1x1
        self.num3x3_reduce = num3x3_reduce
        self.num3x3 = num3x3
        self.num5x5_reduce = num5x5_reduce
        self.num5x5 = num5x5
        self.pool_proj = pool_proj
        super(Inception, self).__init__()

    def _build(self, *args, **kwargs):
        self.block1 = ivy.Sequential(ConvBlock(self.in_channels, self.num1x1, kernel_size=[1, 1], stride=1, padding=0)
                                     )
        self.block2 = ivy.Sequential(ConvBlock(self.in_channels, self.num3x3_reduce, kernel_size=[1, 1], stride=1, padding=0),
                                    ConvBlock(self.num3x3_reduce, self.num3x3, kernel_size=[3, 3], stride=1, padding=1)
                                    )
        self.block3 = ivy.Sequential(ConvBlock(self.in_channels, self.num5x5_reduce, kernel_size=[1, 1], stride=1, padding=0),
                                    # ConvBlock(self.num5x5_reduce, self.num5x5, kernel_size=[5, 5], stride=1, padding="SAME")
                                    ConvBlock(self.num5x5_reduce, self.num5x5, kernel_size=[3, 3], stride=1, padding=1) #1
                                    )
        self.block4 = ivy.Sequential(ivy.MaxPool2D([3,3], 1, 1),
                                    ConvBlock(self.in_channels, self.pool_proj, kernel_size=[1, 1], stride=1, padding=0)
                                    )

    def _forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(x)
        block3 = self.block3(x)
        block4 = self.block4(x)

        return ivy.concat([block1, block2, block3, block4], axis=3)


class Auxiliary(ivy.Module):
    def __init__(self, in_channels, num_classes):
        self.in_channels = in_channels
        self.num_classes = num_classes
        super(Auxiliary, self).__init__()

    def _build(self, *args, **kwargs):
        self.pool = ivy.AdaptiveAvgPool2d([4,4])  # ivy.Shape(1, 4, 4, 512)
        # self.bn = ivy.BatchNorm2D(128, eps=0.001)
        # self.conv = ivy.Conv2D(self.in_channels, 128, [1,1], 1, "SAME", with_bias=False)
        self.conv = ConvBlock(self.in_channels, 128, [1,1], 1, 0)
        self.activation = ivy.ReLU()
        self.fc1 = ivy.Linear(2048, 1024)
        self.dropout = ivy.Dropout(0.7)
        self.fc2 = ivy.Linear(1024, self.num_classes)

    def _forward(self, x):
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        out = self.pool(x)
        out = self.conv(out) # contains weights
        out = ivy.flatten(out, start_dim=1)
        out = self.fc1(out) # contains weights
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out) # contains weights
        return out
