from ivy_models.base import BaseModel, BaseSpec
import ivy
from ivy_models.vit.vit import VisionTransformer
from ivy_models.dino.layers import MultiCropWrapper, DINOHead, DINOBackbone
from ivy_models.vit.layers import partial, ConvStemConfig

class DINOConfig(BaseSpec):
    def __init__(self, image_size: int,
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    in_dim: int = 0,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    num_classes: int = 1000,
    representation_size: ivy.Optional[int] = None,
    norm_layer: ivy.Callable[..., ivy.Module] = partial(ivy.LayerNorm, eps=1e-6),
    conv_stem_configs: ivy.Optional[ivy.List[ConvStemConfig]] = None,
    out_dim: int = 65536,
    use_bn: bool = False,
    norm_last_layer: bool = True,
    nlayers: int = 3,
    hidden_dim_: int = 2048,
    bottleneck_dim: int = 256,
    _weight_init: ivy.Initializer = ivy.GlorotUniform(),
    _bias_init: ivy.Initializer = ivy.Zeros(),
    with_bias: bool = True,
    device=None,
    dtype=None
    ):
        super(DINOConfig, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.in_dim = in_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.conv_stem_configs = conv_stem_configs
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.norm_last_layer = norm_last_layer
        self.nlayers = nlayers
        self.hidden_dim_ = hidden_dim_
        self.bottleneck_dim = bottleneck_dim
        self._weight_init = _weight_init
        self._bias_init = _bias_init
        self.with_bias = with_bias
        self.device = device
        self.dtype = dtype

    def get(self, *attr_names):
        new_dict = {}
        for name in attr_names:
            new_dict[name] = getattr(self, name)
        return new_dict

    def get_vit_attrs(self):
        return self.get(
            "image_size",
            "patch_size",
            "num_layers",
            "num_heads",
            "hidden_dim",
            "mlp_dim",
            "dropout",
            "attention_dropout",
            "num_classes",
            "representation_size",
            "norm_layer",
            "conv_stem_configs"
        )

    def get_head_attrs(self):
        return self.get(
            "in_dim",
            "out_dim",
            "use_bn",
            "norm_last_layer",
            "nlayers",
            "hidden_dim_",
            "bottleneck_dim",
            "_weight_init",
            "_bias_init",
            "with_bias"
        )

class DINONet(BaseModel):

    def __init__(
            self,
            config: DINOConfig,
            v: ivy.Container = None,
    ) -> None:
        self.config = config
        super(DINONet, self).__init__(v=v)

    @classmethod
    def get_spec_class(self):
        return DINOConfig

    def _build(self):
        self.student = DINOBackbone(**self.config.get_vit_attrs())
        self.teacher = DINOBackbone(**self.config.get_vit_attrs())
        self.config.in_dim = self.config.hidden_dim * self.config.num_heads
        self.teacher_head = DINOHead(**self.config.get_head_attrs())
        self.student_head = DINOHead(**self.config.get_head_attrs())

    def _forward(self, x):
        return {
            "student_output": self.student_head(self.student),
            "teacher_output": self.teacher_head(self.teacher)
        }


def dino_base(pretrained=False):
    # instantiate the hyperparameters same as bert
    # set the dropout rate to 0.0 to avoid stochasticity in the output
    config = DINOConfig(
        image_size = 224, patch_size=16, num_layers=12, num_heads=12, hidden_dim=768, mlp_dim=3072,
    )
    model = DINONet(config)
    return model

