# global
import ivy
import ivy_models
from layers import ConvBlock, Inception, Auxiliary


class GoogLeNet(ivy.Module):
    """
    Inception-V1 (GoogLeNet) architecture.

    Args::
        num_classes (int): Number of output classes. Defaults to 1000.
        base_width (int): The base width of the ResNet. Defaults to 64.
        v (ivy.Container): Unused parameter. Can be ignored.

    """

    def __init__(self, num_classes=1000, v: ivy.Container = None,):
        if v is not None:
            self.v = v
        self.num_classes = num_classes
        super(GoogLeNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.conv1 = ConvBlock(3, 64, [7,7], 2, padding=3)
        self.pool1 = ivy.MaxPool2D([3,3], 2, 0)
        self.conv2 = ConvBlock(64, 64, [1,1], 1, padding=0)
        self.conv3 = ConvBlock(64, 192, [3,3], 1, padding=1)
        self.pool3 = ivy.MaxPool2D([3,3], 2, 0)
        self.inception3A = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3B = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool4 = ivy.MaxPool2D([3,3], 2, 0)

        self.inception4A = Inception(480, 192, 96, 208, 16, 48, 64)

        # ivy.flatten()
        self.aux4A = Auxiliary(512, self.num_classes)

        self.inception4B = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4C = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4D = Inception(512, 112, 144, 288, 32, 64, 64)

        self.aux4D = Auxiliary(528, self.num_classes)

        self.inception4E = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool5 = ivy.MaxPool2D([2,2], 2, 0)

        self.inception5A = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5B = Inception(832, 384, 192, 384, 48, 128,128)
        self.pool6 = ivy.AdaptiveAvgPool2d([1,1]) # ((1, 1))

        # ivy.flatten()
        self.dropout = ivy.Dropout(0.2)
        self.fc = ivy.Linear(1024, self.num_classes)


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
        
        
#         out = ivy.reshape(out, (1, 1024, 7, 7))
        out = ivy.permute_dims(out, (0, 3, 1, 2))
        out = self.pool6(out)
        
        out = ivy.flatten(out, start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)

        return out, aux1, aux2


def _inceptionNet_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv/weight"]
    new_mapping = new_key
    if any([kc in old_key for kc in W_KEY]):
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