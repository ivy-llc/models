import ivy
class DoubleConv(ivy.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = ivy.Sequential(
            ivy.Conv2D(in_channels, mid_channels, [3, 3], 1, 1, with_bias=False),
            ivy.BatchNorm2D(mid_channels),
            ivy.ReLU(),
            ivy.Conv2D(mid_channels, out_channels, [3, 3], 1, 1, with_bias=False),
            ivy.BatchNorm2D(out_channels),
            ivy.ReLU()
        )
        super().__init__()

    def _forward(self, x):
        return self.double_conv(x)
    
class Down(ivy.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        self.maxpool_conv = ivy.Sequential(
            ivy.MaxPool2D(2, 2 , 0),
            DoubleConv(in_channels, out_channels)
        )
        super().__init__()

    def _forward(self, x):
        return self.maxpool_conv(x)
    
class Up(ivy.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = ivy.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = ivy.Conv2DTranspose(in_channels, in_channels // 2, [2, 2], 2, "VALID")
            self.conv = DoubleConv(in_channels, out_channels)
        super().__init__()

    def _forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BHWC
        diffY = x2.size()[1] - x1.size()[1]
        diffX = x2.size()[2] - x1.size()[2]

        x1 = ivy.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = ivy.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(ivy.Module):
    def __init__(self, in_channels, out_channels):
        self.conv = ivy.Conv2D(in_channels, out_channels, [1, 1], 1, 0)
        super(OutConv, self).__init__()

    def _forward(self, x):
        return self.conv(x)
    
