import ivy
from ivy_models.convnext.layers import ConvNeXtBlock, ConvNeXtV2Block, ConvNeXtLayerNorm

from ivy_models.base import BaseModel, BaseSpec
from ivy_models.helpers import load_torch_weights


class ConvNeXtSpec(BaseSpec):
    def __init__(
        self,
        version=1,
        in_channels=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        device=None,
        training=False,
    ):
        assert version == 1 or version == 2
        super(ConvNeXtSpec, self).__init__(
            version=version,
            in_channels=in_channels,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            head_init_scale=head_init_scale,
            device=device,
            training=training,
        )


class ConvNeXt(BaseModel):
    def __init__(
        self,
        version=1,
        in_channels=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        device=None,
        training=False,
        spec=None,
        v=None,
    ):
        self.spec = (
            spec
            if spec and isinstance(spec, ConvNeXtSpec)
            else ConvNeXtSpec(
                version=version,
                in_channels=in_channels,
                num_classes=num_classes,
                depths=depths,
                dims=dims,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                head_init_scale=head_init_scale,
                device=device,
                training=training,
            )
        )
        super(ConvNeXt, self).__init__(device=device, v=v)

    def _build(self, *args, **kwargs):
        self.downsample_layers = []
        stem = ivy.Sequential(
            ivy.Conv2D(
                self.spec.in_channels,
                self.spec.dims[0],
                [4, 4],
                (4, 4),
                0,
                data_format="NCHW",
            ),
            ConvNeXtLayerNorm(
                self.spec.dims[0], eps=1e-6, data_format="channels_first"
            ),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = ivy.Sequential(
                ConvNeXtLayerNorm(
                    self.spec.dims[i], eps=1e-6, data_format="channels_first"
                ),
                ivy.Conv2D(
                    self.spec.dims[i],
                    self.spec.dims[i + 1],
                    [2, 2],
                    (2, 2),
                    0,
                    data_format="NCHW",
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = []
        dp_rates = [
            x for x in ivy.linspace(0, self.spec.drop_path_rate, sum(self.spec.depths))
        ]
        cur = 0
        for i in range(4):
            stage = ivy.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=self.spec.dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=self.spec.layer_scale_init_value,
                    )
                    if self.spec.version == 1
                    else ConvNeXtV2Block(
                        dim=self.spec.dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=self.spec.layer_scale_init_value,
                    )
                    for j in range(self.spec.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += self.spec.depths[i]

        self.norm = ivy.LayerNorm([self.spec.dims[-1]], eps=1e-6)
        self.head = ivy.Linear(self.spec.dims[-1], self.spec.num_classes)

        self.norm = ivy.LayerNorm(self.dims[-1], eps=1e-6)
        self.head = ivy.Linear(self.dims[-1], self.num_classes)

    @classmethod
    def get_spec_class(self):
        return ConvNeXtSpec

    def _forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(ivy.mean(x, axis=(-2, -1)))
        x = self.head(x)
        return x


def _convnext_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    if "downsample_layers" in old_key:
        if "0/0/bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}
        elif "0/0/weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b"}
        elif "downsample_layers/0" not in old_key and "1/bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}
        elif "downsample_layers/0" not in old_key and "1/w" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    elif "dwconv" in old_key:
        if "bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}
        elif "weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "a 1 c d -> c d a"}
    elif "grn" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "1 1 1 h -> h"}
    return new_mapping


def convnext_tiny(pretrained=True):
    """Loads a ConvNeXt with specified size, optionally pretrained."""
    depths, dims = ([3, 3, 9, 3], [96, 192, 384, 768])

    if not pretrained:
        return ConvNeXt(version=1, depths=depths, dims=dims)

    weight_url = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"

    model = ConvNeXt(version=1, depths=depths, dims=dims)
    w_clean = load_torch_weights(
        weight_url, model, custom_mapping=_convnext_torch_weights_mapping
    )

    return ConvNeXt(version=1, depths=depths, dims=dims, v=w_clean)


def convnext_small(pretrained=True):
    """Loads a ConvNeXt with specified size, optionally pretrained."""
    depths, dims = ([3, 3, 27, 3], [96, 192, 384, 768])

    if not pretrained:
        return ConvNeXt(version=1, depths=depths, dims=dims)

    weight_url = "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"

    model = ConvNeXt(version=1, depths=depths, dims=dims)
    w_clean = load_torch_weights(
        weight_url, model, custom_mapping=_convnext_torch_weights_mapping
    )
    model.v = w_clean
    return model


def convnext_base(pretrained=True):
    """Loads a ConvNeXt with specified size, optionally pretrained."""
    depths, dims = ([3, 3, 27, 3], [128, 256, 512, 1024])

    if not pretrained:
        return ConvNeXt(version=1, depths=depths, dims=dims)

    weight_url = "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth"

    model = ConvNeXt(version=1, depths=depths, dims=dims)
    w_clean = load_torch_weights(
        weight_url, model, custom_mapping=_convnext_torch_weights_mapping
    )
    model.v = w_clean
    return model


def convnext_large(pretrained=True):
    """Loads a ConvNeXt with specified size, optionally pretrained."""
    depths, dims = ([3, 3, 27, 3], [192, 384, 768, 1536])

    if not pretrained:
        return ConvNeXt(version=1, depths=depths, dims=dims)

    weight_url = "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth"

    model = ConvNeXt(version=1, depths=depths, dims=dims)
    w_clean = load_torch_weights(
        weight_url, model, custom_mapping=_convnext_torch_weights_mapping
    )
    model.v = w_clean
    return model


def convnextv2_atto(pretrained=True):
    """Loads a ConvNeXtV2 with specified size, optionally pretrained."""
    depths, dims = ([2, 2, 6, 2], [40, 80, 160, 320])

    if not pretrained:
        return ConvNeXt(version=2, depths=depths, dims=dims)

    weight_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt"

    model = ConvNeXt(version=2, depths=depths, dims=dims)
    w_clean = load_torch_weights(
        weight_url, model, custom_mapping=_convnext_torch_weights_mapping
    )
    model.v = w_clean
    return model


def convnextv2_base(pretrained=True):
    """Loads a ConvNeXtV2 with specified size, optionally pretrained."""
    depths, dims = ([3, 3, 27, 3], [128, 256, 512, 1024])

    if not pretrained:
        return ConvNeXt(version=2, depths=depths, dims=dims)

    weight_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt"

    model = ConvNeXt(version=2, depths=depths, dims=dims)
    w_clean = load_torch_weights(
        weight_url, model, custom_mapping=_convnext_torch_weights_mapping
    )
    model.v = w_clean
    return model
