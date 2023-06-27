from .unet_layers import *
from ivy_models.helpers import load_torch_weights
import ivy
import builtins


class UNET(ivy.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, v=None):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (UNetDoubleConv(n_channels, 64))
        self.down1 = (UNetDown(64, 128))
        self.down2 = (UNetDown(128, 256))
        self.down3 = (UNetDown(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (UNetDown(512, 1024 // factor))
        self.up1 = (UNetUp(1024, 512 // factor, bilinear))
        self.up2 = (UNetUp(512, 256 // factor, bilinear))
        self.up3 = (UNetUp(256, 128 // factor, bilinear))
        self.up4 = (UNetUp(128, 64, bilinear))
        self.outc = (UNetOutConv(64, n_classes))

        super(UNET, self).__init__(v=v)

    def _forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
def _unet_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    W_KEY = [
        "0/weight",
        "3/weight",
        "conv/weight",
    ]

    if "up/weight" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w b c"}
    elif builtins.any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> w h c b"}
    elif "conv/bias" in old_key or "up/bias" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "h -> 1 1 1 h"}
    
    return new_mapping


def UNet(n_channels=3, n_classes=1, v=None, pretrained=False):
    """UNET model"""
    if not pretrained:
        return UNET(n_channels=n_channels, n_classes=n_classes, v=v)

    reference_model = UNET(n_channels=n_channels, n_classes=n_classes)
    url = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth"
    w_clean = load_torch_weights(
        url, reference_model,raw_keys_to_prune=["num_batches_tracked"] , custom_mapping=_unet_torch_weights_mapping
    )
    return UNET(n_channels=n_channels, n_classes=n_classes, v=w_clean)
