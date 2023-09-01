import ivy


class InceptionConvBlock(ivy.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super(InceptionConvBlock, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv = ivy.Conv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            with_bias=False,
            data_format="NCHW",
        )
        self.bn = ivy.BatchNorm2D(
            self.out_channels, eps=0.001, data_format="NCS", training=False
        )

    def _forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = ivy.relu(x)
        return x


class InceptionBlock(ivy.Module):
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
        super(InceptionBlock, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv_1x1 = InceptionConvBlock(
            self.in_channels, self.num1x1, kernel_size=[1, 1], stride=1, padding=0
        )

        self.conv_3x3 = InceptionConvBlock(
            self.in_channels,
            self.num3x3_reduce,
            kernel_size=[1, 1],
            stride=1,
            padding=0,
        )
        self.conv_3x3_red = InceptionConvBlock(
            self.num3x3_reduce, self.num3x3, kernel_size=[3, 3], stride=1, padding=1
        )

        self.conv_5x5 = InceptionConvBlock(
            self.in_channels,
            self.num5x5_reduce,
            kernel_size=[1, 1],
            stride=1,
            padding=0,
        )
        self.conv_5x5_red = InceptionConvBlock(
            self.num5x5_reduce, self.num5x5, kernel_size=[3, 3], stride=1, padding=1
        )

        self.pool_proj_conv = InceptionConvBlock(
            self.in_channels, self.pool_proj, kernel_size=[1, 1], stride=1, padding=0
        )

    def _forward(self, x):
        # 1x1
        conv_1x1 = self.conv_1x1(x)

        # 3x3
        conv_3x3 = self.conv_3x3(x)
        conv_3x3_red = self.conv_3x3_red(conv_3x3)

        # 5x5
        conv_5x5 = self.conv_5x5(x)
        conv_5x5_red = self.conv_5x5_red(conv_5x5)

        # pool_proj
        pool_proj = ivy.max_pool2d(x, [3, 3], 1, 1, ceil_mode=True, data_format="NCHW")
        pool_proj = self.pool_proj_conv(pool_proj)

        ret = ivy.concat([conv_1x1, conv_3x3_red, conv_5x5_red, pool_proj], axis=1)
        return ret


class InceptionAuxiliaryBlock(ivy.Module):
    def __init__(self, in_channels, num_classes, aux_dropout=0.7):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.aux_dropout = aux_dropout
        super(InceptionAuxiliaryBlock, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv = InceptionConvBlock(self.in_channels, 128, [1, 1], 1, 0)
        self.fc1 = ivy.Linear(2048, 1024, with_bias=True)
        self.dropout = ivy.Dropout(self.aux_dropout)
        self.fc2 = ivy.Linear(1024, self.num_classes, with_bias=True)
        self.softmax = ivy.Softmax()

    def _forward(self, x):
        out = ivy.adaptive_avg_pool2d(x, [4, 4])
        out = self.conv(out)
        out = ivy.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = ivy.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
