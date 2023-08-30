import ivy
from ivy_models.base import BaseSpec, BaseModel


class MLPMixerSpec(BaseSpec):
    def __init__(
        self,
        vol=72,
        inp_dim=3,
        num_classes=10,
        num_blocks=2,
        patch_size=16,
        hidden_dim=128,
        token_mlp_dim=64,
        channel_mlp_dim=32,
        data_format="NHWC",
    ):
        super(MLPMixerSpec, self).__init__(
            vol=vol,
            inp_dim=inp_dim,
            num_classes=num_classes,
            num_blocks=num_blocks,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            token_mlp_dim=token_mlp_dim,
            channel_mlp_dim=channel_mlp_dim,
            data_format=data_format,
        )


class MLPBlock(ivy.Module):
    def __init__(self, inp_dim, mlp_dim):
        self.linear0 = ivy.Linear(inp_dim, mlp_dim)
        self.gelu = ivy.GELU()
        self.linear1 = ivy.Linear(mlp_dim, inp_dim)
        self.dropout = ivy.Dropout(0.25)
        super(MLPBlock, self).__init__()

    def _forward(self, x):
        y = self.linear0(x)
        y = self.gelu(y)
        y = self.linear1(y)
        return y


class MixerBlock(ivy.Module):
    def __init__(self, hidden_dim, num_patches, token_mlp_dim, channel_mlp_dim):
        self.norm1 = ivy.LayerNorm([hidden_dim])
        self.mlp_block1 = MLPBlock(num_patches, token_mlp_dim)
        self.norm2 = ivy.LayerNorm([hidden_dim])
        self.mlp_block2 = MLPBlock(hidden_dim, channel_mlp_dim)
        super(MixerBlock, self).__init__()

    def _forward(self, x):
        y = self.norm1(x)
        y = ivy.matrix_transpose(y)
        y = self.mlp_block1(y)
        y = ivy.matrix_transpose(y)
        x = x + y
        y = self.norm2(x)
        y = x + self.mlp_block2(y)
        return y


class MLPMixer(BaseModel):
    def __init__(
        self,
        vol=72,
        inp_dim=3,
        num_classes=10,
        num_blocks=2,
        patch_size=16,
        hidden_dim=128,
        token_mlp_dim=64,
        channel_mlp_dim=32,
        data_format="NHWC",
        spec=None,
        v=None,
    ):
        self.spec = (
            spec
            if spec and isinstance(spec, MLPMixerSpec)
            else MLPMixerSpec(
                vol=vol,
                inp_dim=inp_dim,
                num_classes=num_classes,
                num_blocks=num_blocks,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim,
                data_format=data_format,
            )
        )

        super(MLPMixer, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.conv = ivy.Conv2D(
            self.spec.inp_dim,
            self.spec.hidden_dim,
            [self.spec.patch_size, self.spec.patch_size],
            [self.spec.patch_size, self.spec.patch_size],
            0,
            data_format="NHWC",
        )
        self.num_patches = int((self.spec.vol / self.spec.patch_size) ** 2)
        self.mixer_block = MixerBlock(
            self.spec.hidden_dim,
            self.num_patches,
            self.spec.token_mlp_dim,
            self.spec.channel_mlp_dim,
        )
        self.norm = ivy.LayerNorm([self.spec.hidden_dim])
        self.dropout = ivy.Dropout(0.25)
        self.linear = ivy.Linear(self.spec.hidden_dim, self.spec.num_classes)

    @classmethod
    def get_spec_class(self):
        return MLPMixerSpec

    def _forward(self, x, data_format=None):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NCHW":
            x = ivy.permute_dims(x, (0, 2, 3, 1))
        x = self.conv(x)
        x = x.reshape(
            (int(x.shape[0]), int(x.shape[1]) * int(x.shape[2]), int(x.shape[3]))
        )
        for i in range(self.spec.num_blocks):
            x = self.mixer_block(x)
        x = self.norm(x)
        x = ivy.mean(x, axis=1)
        logits = self.linear(x)
        probs = ivy.softmax(logits)

        return probs


def mlpmixer(
    pretrained=True,
    vol=72,
    inp_dim=3,
    num_classes=10,
    num_blocks=4,
    patch_size=9,
    hidden_dim=128,
    token_mlp_dim=64,
    channel_mlp_dim=128,
    data_format="NHWC",
):
    """Ivy MLPMixer model"""
    if pretrained:
        return MLPMixer.load_from_huggingface(
            repo_id="unifyai/MLPMixer", weights_path="weights.hdf5"
        )
    model = MLPMixer(
        vol=vol,
        inp_dim=inp_dim,
        num_classes=num_classes,
        num_blocks=num_blocks,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        token_mlp_dim=token_mlp_dim,
        channel_mlp_dim=channel_mlp_dim,
        data_format=data_format,
    )
    return model
