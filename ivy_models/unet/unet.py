import ivy
from ivy_models.unet.layers import UNetDoubleConv, UNetDown, UNetOutConv, UNetUp

import builtins
from ivy_models.helpers import load_torch_weights
from ivy_models.base import BaseSpec, BaseModel


class UNetSpec(BaseSpec):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetSpec, self).__init__(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=bilinear,
        )


class UNET(BaseModel):
    def __init__(self, n_channels, n_classes, bilinear=False, spec=None, v=None):
        self.spec = (
            spec
            if spec and isinstance(spec, UNetSpec)
            else UNetSpec(n_channels, n_classes, bilinear)
        )
        super(UNET, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.factor = 2 if self.spec.bilinear else 1
        self.inc = UNetDoubleConv(self.spec.n_channels, 64)
        self.down1 = UNetDown(64, 128)
        self.down2 = UNetDown(128, 256)
        self.down3 = UNetDown(256, 512)
        self.down4 = UNetDown(512, 1024 // self.factor)
        self.up1 = UNetUp(1024, 512 // self.factor, self.spec.bilinear)
        self.up2 = UNetUp(512, 256 // self.factor, self.spec.bilinear)
        self.up3 = UNetUp(256, 128 // self.factor, self.spec.bilinear)
        self.up4 = UNetUp(128, 64, self.spec.bilinear)
        self.outc = UNetOutConv(64, self.spec.n_classes)

    @classmethod
    def get_spec_class(self):
        return UNetSpec

    def _forward(self, x, data_format="NHWC"):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NCHW":
            x = ivy.permute_dims(x, (0, 2, 3, 1))
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
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    elif "conv/bias" in old_key or "up/bias" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "h -> 1 1 1 h"}

    return new_mapping


def unet_carvana(n_channels=3, n_classes=2, v=None, pretrained=True):
    model = UNET(n_channels=3, n_classes=2)
    if pretrained:
        url = "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth"  # noqa
        w_clean = load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_unet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


if __name__ == "__main__":
    ivy.set_torch_backend()
    model = unet_carvana()
