from ivy_models.helpers import load_torch_weights
from ivy_models.vit.layers import (
    Callable,
    Conv2dNormActivation,
    ConvStemConfig,
    List,
    Optional,
    VIT_Encoder,
    Zeros,
    ivy,
    partial,
)
from ivy_models.base import BaseModel, BaseSpec


class VisionTransformerSpec(BaseSpec):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., ivy.Module] = partial(ivy.LayerNorm, eps=1e-6),
        data_format: str="NHWC",
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        ivy.utils.assertions.check_true(
            image_size % patch_size == 0, "Input shape indivisible by patch size!"
        )

        super(VisionTransformerSpec, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            representation_size=representation_size,
            norm_layer=norm_layer,
            data_format=data_format,
            conv_stem_configs=conv_stem_configs,
        )


class VisionTransformer(BaseModel):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., ivy.Module] = partial(ivy.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        data_format: str="NHWC",
        spec=None,
        v=None,
    ):
        self.spec = (
            spec
            if spec and isinstance(spec, VisionTransformerSpec)
            else VisionTransformerSpec(
                image_size=image_size,
                patch_size=patch_size,
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                num_classes=num_classes,
                representation_size=representation_size,
                norm_layer=norm_layer,
                data_format=data_format,
                conv_stem_configs=conv_stem_configs,
            )
        )
        super().__init__(v=v)

    def _build(self, *args, **kwargs):
        if self.spec.conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = []
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(self.spec.conv_stem_configs):
                seq_proj.append(Conv2dNormActivation(
                    in_channels=prev_channels,
                    out_channels=conv_stem_layer_config.out_channels,
                    kernel_size=conv_stem_layer_config.kernel_size,
                    stride=conv_stem_layer_config.stride,
                    norm_layer=conv_stem_layer_config.norm_layer,
                    activation_layer=conv_stem_layer_config.activation_layer,
                ))
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.append(ivy.Conv2D(
                prev_channels, self.spec.hidden_dim, [1, 1], 1, 0
            ))
            self.conv_proj: ivy.Module = ivy.Sequential(*seq_proj)
        else:
            self.conv_proj = ivy.Conv2D(
                3,
                self.spec.hidden_dim,
                [self.spec.patch_size, self.spec.patch_size],
                self.spec.patch_size,
                0,
            )

        seq_length = (self.spec.image_size // self.spec.patch_size) ** 2

        # Add a class token
        self._class_token_shape = (1, 1, self.spec.hidden_dim)
        self.class_token = Zeros()
        seq_length += 1

        self.encoder = VIT_Encoder(
            seq_length,
            self.spec.num_layers,
            self.spec.num_heads,
            self.spec.hidden_dim,
            self.spec.mlp_dim,
            self.spec.dropout,
            self.spec.attention_dropout,
            self.spec.norm_layer,
        )
        self.seq_length = seq_length

        heads_layers = []
        if self.spec.representation_size is None:
            heads_layers.append(ivy.Linear(
                self.spec.hidden_dim, self.spec.num_classes
            ))
        else:
            heads_layers.append(ivy.Linear(
                self.spec.hidden_dim, self.spec.representation_size
            ))
            heads_layers.append(ivy.tanh())
            heads_layers.append(ivy.Linear(
                self.spec.representation_size, self.spec.num_classes
            ))

        self.heads = ivy.Sequential(*heads_layers)

    def _create_variables(self, device, dtype=None):
        return {
            "class_token": self.class_token.create_variables(
                self._class_token_shape, device, dtype=dtype
            )
        }

    def _process_input(self, x):
        n, h, w, c = x.shape
        p = self.spec.patch_size
        ivy.utils.assertions.check_true(
            h == self.spec.image_size,
            f"Wrong image height! Expected {self.spec.image_size} but got {h}!",
        )
        ivy.utils.assertions.check_true(
            w == self.spec.image_size,
            f"Wrong image width! Expected {self.spec.image_size} but got {w}!",
        )
        n_h = h // p
        n_w = w // p

        # (n, h, w, c) -> (n, n_h, n_w, self.hidden_dim)
        x = self.conv_proj(x)
        # (n, n_h, n_w, self.hidden_dim) -> (n, (n_h * n_w), self.hidden_dim)
        x = x.reshape(shape=(n, n_h * n_w, self.spec.hidden_dim))

        return x

    @classmethod
    def get_spec_class(self):
        return VisionTransformerSpec

    def _forward(self, x, data_format: str="NHWC"):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NCHW":
            x = ivy.permute_dims(x, (0, 2, 3, 1))
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.v.class_token.expand((n, -1, -1))
        x = ivy.concat([batch_class_token, x], axis=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def _vit_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key

    if "conv_proj/weight" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    elif "conv_proj/bias" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "h -> 1 1 1 h"}

    return new_mapping


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    data_format: str="NHWC",
    v=None,
) -> VisionTransformer:
    model = VisionTransformer(
        image_size=224,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        data_format=data_format,
        v=v,
    )

    return model


def vit_b_16(data_format="NHWC", pretrained=True) -> VisionTransformer:
    model = _vision_transformer(
        patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, data_format=data_format
    )
    if pretrained:
        url = "https://download.pytorch.org/models/vit_b_16-c867db91.pth"
        w_clean = load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_vit_torch_weights_mapping,
        )
        model.v = w_clean
    return model


def vit_b_32(data_format="NHWC", pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=32, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072, data_format=data_format
    )
    if pretrained:
        url = "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth"
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_vit_torch_weights_mapping,
        )
        ref_model.v = w_clean
    return ref_model


def vit_l_16(data_format="NHWC", pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=16, num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096, data_format=data_format
    )
    if pretrained:
        url = "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth"
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_vit_torch_weights_mapping,
        )
        ref_model.v = w_clean
    return ref_model


def vit_l_32(data_format="NHWC", pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=32, num_layers=24, num_heads=16, hidden_dim=1024, mlp_dim=4096, data_format=data_format
    )
    if pretrained:
        url = "https://download.pytorch.org/models/vit_l_32-c7638314.pth"
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_vit_torch_weights_mapping,
        )
        ref_model.v = w_clean
    return ref_model


def vit_h_14(data_format="NHWC", pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=14, num_layers=32, num_heads=16, hidden_dim=1280, mlp_dim=5120, data_format=data_format
    )
    if pretrained:
        url = "https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth"
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_vit_torch_weights_mapping,
        )
        ref_model.v = w_clean
    return ref_model
