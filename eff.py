import ivy 
import copy
import collections
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., ivy.Module]


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., ivy.Module]] = None,
    ) -> None:
        input_channels = adjust_channels(input_channels, width_mult, 8)
        out_channels = adjust_channels(out_channels, width_mult, 8)
        num_layers = adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)



class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., ivy.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def adjust_channels(a, b, c):
    return _make_divisible(a * b, c)


def adjust_depth(a, b):
    return int(ivy.ceil(a * b))


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(ivy.repeat(x, n))


# def stochastic_depth(input: ivy.Array, p: float, mode: str, training: bool = True) -> ivy.Array:
#     """
#     Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
#     <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
#     branches of residual architectures.

#     Args:
#         input (ivy.Array[N, ...]): The input tensor or arbitrary dimensions with the first one
#                     being its batch i.e. a batch with ``N`` rows.
#         p (float): probability of the input to be zeroed.
#         mode (str): ``"batch"`` or ``"row"``.
#                     ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
#                     randomly selected rows from the batch.
#         training: apply stochastic depth if is ``True``. Default: ``True``

#     Returns:
#         ivy.Array[N, ...]: The randomly zeroed tensor.
#     """
#     if p < 0.0 or p > 1.0:
#         raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
#     if mode not in ["batch", "row"]:
#         raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
#     if not training or p == 0.0:
#         return input

#     survival_rate = 1.0 - p
#     # if mode == "row":
#     #     size = [input.shape[0]] + [1] * (input.ndim - 1)
#     # else:
#     # size = [1] * input.ndim
#     noise = ivy.empty((input.shape[0], 1, 1, 1), dtype=input.dtype, device=input.device)
#     noise = noise.bernoulli(logits=survival_rate)
#     if survival_rate > 0.0:
#         noise.divide(survival_rate)
#     return input * noise


def stochastic_depth(x, p):
    survival_rate = 1.0 - p
    binary_tensor = (
        ivy.random_uniform(
            shape=(x.shape[0], 1, 1, 1), low=0, high=1, device=x.device
        )
        < survival_rate
    )
    return ivy.divide(x, survival_rate) * binary_tensor


class StochasticDepth(ivy.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        self.p = p
        self.mode = mode

        super().__init__()

    def _forward(self, input: ivy.Array) -> ivy.Array:
        return stochastic_depth(input, self.p)


class SqueezeExcitation(ivy.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., ivy.Module] = ivy.ReLU,
        scale_activation: Callable[..., ivy.Module] = ivy.sigmoid,
    ) -> None:
        self.fc1 = ivy.Conv2D(input_channels, squeeze_channels, [1, 1], 1, 0)
        self.fc2 = ivy.Conv2D(squeeze_channels, input_channels, [1, 1], 1, 0)
        self.activation = activation()
        self.scale_activation = scale_activation

        super().__init__()

    def _scale(self, x: ivy.Array) -> ivy.Array:
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = ivy.adaptive_avg_pool2d(x, 1)
        scale = ivy.permute_dims(x, (0, 2, 3, 1))
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def _forward(self, input: ivy.Array) -> ivy.Array:
        scale = self._scale(input)
        return scale * input
    

class Conv2dNormActivation(ivy.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., ivy.Module]] = ivy.BatchNorm2D,
        activation_layer: Optional[Callable[..., ivy.Module]] = ivy.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., ivy.Module] = ivy.Conv2D,
        depthwise: bool = False,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        #         kernel_size = [kernel_size, kernel_size]
        #     else:
        #         _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
        #         kernel_size = _make_ntuple(kernel_size, _conv_dim)
        #         dilation = _make_ntuple(dilation, _conv_dim)
        #         padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None
        if not depthwise:
            layers = [
                conv_layer(
                    in_channels,
                    out_channels,
                    [kernel_size, kernel_size],
                    stride,
                    padding,
                    dilations=dilation,
                    with_bias=bias,
                )
            ]
        else:
            layers = [
                ivy.DepthwiseConv2D(
                    in_channels,
                    [kernel_size, kernel_size],
                    stride,
                    padding,
                    dilations=dilation,
                    with_bias=bias,
                )
            ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        self.out_channels = out_channels


class MBConv(ivy.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., ivy.Module],
        se_layer: Callable[..., ivy.Module] = SqueezeExcitation,
    ) -> None:

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = ivy.SiLU

        # expand
        expanded_channels = adjust_channels(cnf.input_channels, cnf.expand_ratio, 8)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                depthwise=True,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=ivy.SiLU))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = ivy.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

        super().__init__()

    def _forward(self, input: ivy.Array) -> ivy.Array:
        result = self.block(input)
        # if self.use_res_connect:
        #     result = self.stochastic_depth(result)
        #     result += input
        return result


class FusedMBConv(ivy.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., ivy.Module],
    ) -> None:

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = ivy.SiLU

        expanded_channels = adjust_channels(cnf.input_channels, cnf.expand_ratio, 8)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = ivy.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

        super().__init__()

    def _forward(self, input: ivy.Array) -> ivy.Array:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result
    

class EfficientNet(ivy.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., ivy.Module]] = None,
        last_channel: Optional[int] = None,
        v=None,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        if norm_layer is None:
            norm_layer = ivy.BatchNorm2D

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=ivy.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(ivy.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=ivy.SiLU,
            )
        )

        self.features = ivy.Sequential(*layers)
        self.classifier = ivy.Sequential(
            ivy.Dropout(dropout),
            ivy.Linear(lastconv_output_channels, num_classes),
        )

        super(EfficientNet, self).__init__(v=v)

        # if v is not None:
        #     self.v = v

    def _forward_impl(self, x: ivy.Array) -> ivy.Array:
        x = self.features(x)

        x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = ivy.adaptive_avg_pool2d(x, 1)
        x = ivy.permute_dims(x, (0, 2, 3, 1))

        x = ivy.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    def _forward(self, x: ivy.Array) -> ivy.Array:
        return self._forward_impl(x)


if __name__ == "__main__":
    # Preprocess torch image
    import torch
    from torchvision import transforms
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    from PIL import Image

    filename = "images/cat.jpg"
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
    torch_img = Image.open(filename)
    torch_img = preprocess(torch_img)
    torch_img = torch.unsqueeze(torch_img, 0)
    img = torch_img.numpy().reshape(1, 224, 224, 3)

    torch_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    torch_model.eval()
    # res = torch_model(torch_img)

    torch_output = torch.softmax(torch_model(torch_img), dim=1)
    torch_classes = torch.argsort(torch_output[0], descending=True)[:3]
    torch_logits = torch.take(torch_output[0], torch_classes)

    print("Indices of the top 3 classes are:", torch_classes)
    print("Logits of the top 3 classes are:", torch_logits)

    ivy.set_torch_backend()
    torch_v = ivy.Container.cont_from_disk_as_pickled("b0.pickled")
    
    def _rename_v(dictionary):
        for key, value in list(dictionary.items()):
            if isinstance(value, dict):
                _rename_v(value)

            try:
                _ = int(key)
                new_key = "v" + key
                val = dictionary.pop(key)
                dictionary[new_key] = ivy.Container({"submodules": val}) if list(val.keys())[0] not in ["fc1", "fc2", "weight", "bias", "w", "b", "running_mean", "running_var", "block"] else val
            except Exception:
                if key in ["classifier", "features", "block"]:
                    dictionary[key] = ivy.Container({"submodules": dictionary.pop(key)}) 
                elif key == "weight":
                    val = dictionary.pop(key)
                    dictionary["w"] = val["w"] if isinstance(val, dict) else val
                elif key == "bias":
                    val = dictionary.pop(key)
                    dictionary["b"] = val["b"] if isinstance(val, dict) else val

    # def _permute_v(dictionary):
    #     for key, value in dictionary.items():
    #         if isinstance(value, dict):
    #             _permute_v(value)
    #         else:
    #             if len(value.shape) == 4:
    #                 dictionary[key] = ivy.permute_dims(value, (0, 3, 1, 2))
    #             elif len(value.shape) == 3:
    #                 dictionary[key] = ivy.permute_dims(value, (0, 2, 1))
            

    _rename_v(torch_v)
    # _permute_v(torch_v)
    # print(torch_v)
    
    # ivy.set_torch_backend()
    inverted_residual_setting = [
        MBConvConfig(1, 3, 1, 32, 16, 1),
        MBConvConfig(6, 3, 2, 16, 24, 2),
        MBConvConfig(6, 5, 2, 24, 40, 2),
        MBConvConfig(6, 3, 2, 40, 80, 3),
        MBConvConfig(6, 5, 1, 80, 112, 3),
        MBConvConfig(6, 5, 2, 112, 192, 4),
        MBConvConfig(6, 3, 1, 192, 320, 1),
    ]
    last_channel = None
    dropout = 0.2
    ivy_model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, v=torch_v)
    device = "cpu"

    # ivy_model.v = torch_v
    # ret = ivy_model(img)
    # print(ret)
    # print(ivy_model.v)

    # print(torch_v.cont_all_key_chains())

    # print("\n","\n", ivy_model.v.cont_all_key_chains())

    output = ivy.softmax(ivy_model(ivy.asarray(img)))  # pass the image to the model
    classes = ivy.argsort(output[0], descending=True)[:3]  # get the top 3 classes
    logits = ivy.gather(output[0], classes)  # get the logits

    print("Indices of the top 3 classes are:", classes)
    print("Logits of the top 3 classes are:", logits)
