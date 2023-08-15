import ivy
import ivy_models
from ivy_models.base import BaseModel, BaseSpec
from ivy_models.dino.utils import trunc_normal_
from torchvision import transforms
from ivy_models.vit.vit import VisionTransformer
from ivy.stateful.initializers import Initializer, GlorotUniform, Zeros
from ivy_models.vit.layers import partial, ConvStemConfig
from ivy_models_tests.helpers import image_helpers
from PIL import Image

class DINOBackbone(ivy.Module):

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
            representation_size: ivy.Optional[int] = None,
            norm_layer: ivy.Callable[..., ivy.Module] = partial(ivy.LayerNorm, eps=1e-6),
            conv_stem_configs: ivy.Optional[ivy.List[ConvStemConfig]] = None,
            spec=None,
            v: ivy.Container = None,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.conv_stem_configs = conv_stem_configs
        super(DINOBackbone, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.backbone = VisionTransformer(image_size = self.image_size, patch_size=self.patch_size, num_layers=self.num_layers,
                                          num_heads=self.num_heads, hidden_dim=self.hidden_dim, mlp_dim=self.mlp_dim)

    def _forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = ivy.cumsum(ivy.unique_consecutive(
            ivy.array([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx, output = 0, ivy.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(ivy.cat(x[start_idx: end_idx]))

            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = ivy.cat((output, _out))
            start_idx = end_idx
        return output

class DINOHead(ivy.Module):
    """DINO architecture"""
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            use_bn: bool = False,
            norm_last_layer : bool = True,
            nlayers: int = 3,
            hidden_dim_: int = 2048,
            bottleneck_dim: int = 256,
            _weight_init: Initializer = GlorotUniform(),
            _bias_init: Initializer = Zeros(),
            with_bias: bool = True,
            device=None,
            dtype=None,
            v: ivy.Container = None,
        ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.norm_last_layer = norm_last_layer
        self.nlayers = nlayers
        self.hidden_dim_ = hidden_dim_
        self.bottleneck_dim = bottleneck_dim
        self._w_shape = (out_dim, in_dim)
        self._b_shape = (out_dim,)
        self._weight_init = _weight_init
        self._b_init = _bias_init
        self.with_bias = with_bias
        super(DINOHead, self).__init__(v=v, device=device, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        w = self._weight_init.create_variables(
                self._w_shape, device,self.out_dim,
                    self.in_dim, dtype
        )
        v = {
            "w": trunc_normal_(w, std=.02),
        }
        v = dict(
            **v,
            b=self._b_init.create_variables(
                self._b_shape,
                device,
                self.out_dim,
                self.in_dim,
                dtype=dtype,
            ),
        )
        return v

    def _build(self, *args, **kwargs):
        nlayers = max(self.nlayers, 1)
        if nlayers == 1:
            self.mlp = ivy.Linear(self.in_dim, self.bottleneck_dim)
        else:
            layers = [ivy.Linear(self.in_dim, self.bottleneck_dim)]
            # TODO: change back to batchnorm1d when changes are merged
            if self.use_bn:
                layers.append(ivy.BatchNorm2D(self.hidden_dim_))
            layers.append(ivy.GELU())
            for _ in range(nlayers-2):
                layers.append(ivy.Linear(self.hidden_dim_, self.hidden_dim_))
                if self.use_bn:
                    layers.append(ivy.BatchNorm2D(self.hidden_dim_))
                layers.append(ivy.GELU())
            layers.append(ivy.Linear(self.hidden_dim_, self.bottleneck_dim))
            self.mlp = ivy.Sequential(*layers)
        # TODO: weight normalization
        self.last_layer = ivy.Linear(self.bottleneck_dim, self.out_dim)
        print(dir(self.last_layer))
        self.last_layer.v.w = ivy.full_like(self.last_layer.v.w, 1.0)
        # if self.norm_last_layer:
        #     self.last_layer.v.w.requires_grad = False

    # def _init_weights(self, module):
    #     # if isinstance(module, ivy.Linear):
    #     trunc_normal_(module.weight, std=.02)
    #     module.w.data.normal_(mean=0.0, std=.02)
    #     if module.b is not None:
    #         module.b.data.zero_()
    #     return module

    def _forward(self, x):
        x = self.mlp(x)
        x = ivy.functional.lp_normalize(x, p = 2., axis = 1)
        x = self.last_layer(x)
        return x


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crop_1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            image_helpers.GaussianBlur(1.0),
            normalize,
        ])

        self.global_crop_2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale = global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            image_helpers.GaussianBlur(0.1),
            image_helpers.Solarization(0.2),
            normalize,
        ])

        self.local_coprs_number = local_crops_number
        self.local_crop = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            image_helpers.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_crop_1(image))
        crops.append(self.global_crop_2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_crop(image))
        return crops





class MultiCropWrapper(ivy.Module):

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        backbone.fc, backbone.head = ivy.Identity, ivy.Identity
        self.backbone = backbone
        self.head = head


    def _forward(self, x):
        if not isinstance(x, list):
            x = [x]
        idx_crops = ivy.cumsum(ivy.unique_consecutive(
            ivy.array([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)

        start_idx, output = 0, ivy.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(ivy.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = ivy.cat((output, _out))
            start_idx = end_idx
            # Run the head forward on the concatenated features.
        return self.head(output)
