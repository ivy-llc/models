import ivy
import ivy_models
from ivy_models.base import BaseModel, BaseSpec
from utils import trunc_normal_
from torchvision import transforms
from ivy_models_tests.helpers import image_helpers
from PIL import Image

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
        super(DinoHeadSpec, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            use_bn=use_bn,
            norm_last_layer=norm_last_layer,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )

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
        # TODO: add batchnorm 1d instead of 2d
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
            self._init_weights(v)
            # TODO: weight normalization
            self.last_layer = ivy.LayerNorm(ivy.Linear(bottleneck_dim, out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False

        def _init_weights(self, module):
            if isinstance(module, ivy.Linear):
                # trunc_normal_(module.weight, std=.02)
                module.weight.data.normal_(mean=0.0, std=.02)
                if isinstance(module, ivy.Linear) and module.bias is not None:
                    module.bias.data.zero_()

        def _forward(self, x):
            x = self.mlp(x)
            x = ivy.functional.lp_normalize(x, p = 2., axis = 1)
            x = self.last_layer(x)
            return x

#TODO: DINOLOSS losses functional not here
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
