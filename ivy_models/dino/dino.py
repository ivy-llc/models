import ivy
import ivy_models
from ivy_models.base import BaseModel, BaseSpec
from utils import trunc_normal_

class DinoHeadSpec(BaseSpec):
    """Dino Head Spec Class"""
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            use_bn: bool = False,
            norm_last_layer: bool = True,
            nlayers: int = 3,
            hidden_dim: int = 2048,
            bottleneck_dim: int = 256,
        ) -> None:
        pass

class DinoHead(BaseModel):
    """DINO architecture"""
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            use_bn: bool = False,
            norm_last_layer : bool = True,
            nlayers: int = 3,
            hidden_dim: int = 2048,
            bottleneck_dim: int = 256,
            spec = None,
            v: ivy.Container = None,
        ) -> None:
        self.spec = (
            spec
            if spec and isinstance(spec, DinoHeadSpec)
            else DinoHeadSpec(
                in_dim, out_dim, use_bn, norm_last_layer, nlayers, hidden_dim, bottleneck_dim
            )
        )
        super(DinoHead, self).__init__(v=v)

        def _build(self, *args, **kwargs):
            nlayers = max(self.nlayers, 1)
            if nlayers ==1:
                self.mlp = ivy.Linear(in_dim, bottleneck_dim)
            else:
                layers = [ivy.Linear(in_dim, bottleneck_dim)]
                if use_bn:
                    layers.append(ivy.BatchNorm2D(hidden_dim))
                layers.append(ivy.GELU())
                for _ in range(nlayers-2):
                    layers.append(ivy.Linear(hidden_dim, hidden_dim))
                    if use_bn:
                        layers.append(ivy.BatchNorm2D(hidden_dim))
                    layers.append(ivy.GELU())
                layers.append(ivy.Linear(hidden_dim, bottleneck_dim))
                self.mlp = ivy.Sequential(*layers)
            self.apply(self._init_weights)
            self.last_layer = nn.utils.weight_norm(ivy.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False

        def _init_weights(self, module):
            if isinstance(module, ivy.Linear):
                ivy.trunc()
                # trunc_normal_(module.weight, std=.02)
                module.weight.data.normal_(mean=0.0, std=.02)
                if isinstance(module, ivy.Linear) and module.bias is not None:
                    module.bias.data.zero_()

        def _forward(self, x):
            x = self.mlp(x)
            x = ivy.functional.lp_normalize(x, p = 2., axis = 1)
            x = self.last_layer(x)
            return x

