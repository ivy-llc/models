# global
import builtins
import ivy
import ivy_models


# Building the initial Convolutional Block
class ConvBlock(ivy.Module):
    def __init__(self, in_chaivyels, out_chaivyels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = ivy.Conv2d(in_chaivyels, out_chaivyels, kernel_size, stride, padding)
        self.bn = ivy.BatchNorm2d(out_chaivyels)
        self.activation = ivy.ReLU()

    def forward(self, x):
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
        super(Inception, self).__init__()

        # Four output channel for each parallel block of network
        # Note, within Inception the individual blocks are running parallely
        # NOT sequentially.
        self.block1 = ivy.Sequential(
            ConvBlock(in_channels, num1x1, kernel_size=1, stride=1, padding=0)
        )

        self.block2 = ivy.Sequential(
            ConvBlock(in_channels, num3x3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num3x3_reduce, num3x3, kernel_size=3, stride=1, padding=1),
        )

        self.block3 = ivy.Sequential(
            ConvBlock(in_channels, num5x5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(num5x5_reduce, num5x5, kernel_size=5, stride=1, padding=2),
        )

        self.block4 = ivy.Sequential(
            ivy.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            ConvBlock(in_channels, pool_proj, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        # Note the different way this forward function
        # calculates the output.
        block1 = self.block1(x)
        block2 = self.block2(x)
        block3 = self.block3(x)
        block4 = self.block4(x)

        return ivy.concat([block1, block2, block3, block4], 1)

class Auxiliary(ivy.Module):
    def __init__(self, in_channels, num_classes):
        super(Auxiliary, self).__init__()

        self.pool = ivy.AdaptiveAvgPool2d((4, 4))
        self.conv = ivy.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.activation = ivy.ReLU()

        self.fc1 = ivy.Linear(2048, 1024)
        self.dropout = ivy.Dropout(0.7)
        self.fc2 = ivy.Linear(1024, num_classes)

    def forward(self, x):
        out = self.pool(x)

        out = self.conv(out)
        out = self.activation(out)
        print('out shape is  ', out.shape)
        # out shape is  torch.Size([2, 128, 4, 4])

        out = ivy.flatten(out, 1)

        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out

class GoogLeNet(ivy.Module):
    def __init__(self, num_classes=1000, v=None):
        super(GoogLeNet, self).__init__()
        if v is not None:
            self.v = v
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = ivy.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)
        self.conv2 = ConvBlock(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool3 = ivy.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception3A = Inception(
            in_channels=192,
            num1x1=64,
            num3x3_reduce=96,
            num3x3=128,
            num5x5_reduce=16,
            num5x5=32,
            pool_proj=32,
        )
        self.inception3B = Inception(
            in_channels=256,
            num1x1=128,
            num3x3_reduce=128,
            num3x3=192,
            num5x5_reduce=32,
            num5x5=96,
            pool_proj=64,
        )
        self.pool4 = ivy.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception4A = Inception(
            in_channels=480,
            num1x1=192,
            num3x3_reduce=96,
            num3x3=208,
            num5x5_reduce=16,
            num5x5=48,
            pool_proj=64,
        )
        self.inception4B = Inception(
            in_channels=512,
            num1x1=160,
            num3x3_reduce=112,
            num3x3=224,
            num5x5_reduce=24,
            num5x5=64,
            pool_proj=64,
        )
        self.inception4C = Inception(
            in_channels=512,
            num1x1=128,
            num3x3_reduce=128,
            num3x3=256,
            num5x5_reduce=24,
            num5x5=64,
            pool_proj=64,
        )
        self.inception4D = Inception(
            in_channels=512,
            num1x1=112,
            num3x3_reduce=144,
            num3x3=288,
            num5x5_reduce=32,
            num5x5=64,
            pool_proj=64,
        )
        self.inception4E = Inception(
            in_channels=528,
            num1x1=256,
            num3x3_reduce=160,
            num3x3=320,
            num5x5_reduce=32,
            num5x5=128,
            pool_proj=128,
        )
        self.pool5 = ivy.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.inception5A = Inception(
            in_channels=832,
            num1x1=256,
            num3x3_reduce=160,
            num3x3=320,
            num5x5_reduce=32,
            num5x5=128,
            pool_proj=128,
        )
        self.inception5B = Inception(
            in_channels=832,
            num1x1=384,
            num3x3_reduce=192,
            num3x3=384,
            num5x5_reduce=48,
            num5x5=128,
            pool_proj=128,
        )
        self.pool6 = ivy.adaptive_avg_pool2d((1, 1))

        self.dropout = ivy.Dropout(0.4)
        self.fc = ivy.Linear(1024, num_classes)

        self.aux4A = Auxiliary(512, num_classes)
        self.aux4D = Auxiliary(528, num_classes)

    def _forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.inception3A(out)
        out = self.inception3B(out)
        out = self.pool4(out)
        out = self.inception4A(out)

        aux1 = self.aux4A(out)

        out = self.inception4B(out)
        out = self.inception4C(out)
        out = self.inception4D(out)

        aux2 = self.aux4D(out)

        out = self.inception4E(out)
        out = self.pool5(out)
        out = self.inception5A(out)
        out = self.inception5B(out)
        out = self.pool6(out)
        out = ivy.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)

        # we need all 3 because loss func of inceptionNet
        # is weighted avg of these 3 out, aux1, and aux2
        return out, aux1, aux2

def _inceptionNet_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv1/weight", "conv2/weight", "downsample/0/weight"]
    new_mapping = new_key
    if builtins.any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping

def inceptionNet_v1(pretrained=True):
    """InceptionNet-V1 model"""
    if not pretrained:
        return GoogLeNet()

    reference_model = GoogLeNet()
    url = "https://download.pytorch.org/models/googlenet-1378be20.pth"
    w_clean = ivy_models.helpers.load_torch_weights(
        url,
        reference_model,
        raw_keys_to_prune=["num_batches_tracked"],
        custom_mapping=_inceptionNet_torch_weights_mapping,
    )
    return GoogLeNet(v=w_clean)

