from collections import OrderedDict
import ivy


class IvyGELUTanh(ivy.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: ivy.Array) -> ivy.Array:
        return ivy.gelu(input, approximate="tanh")


class NewGELUActivation(ivy.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: ivy.Array) -> ivy.Array:
        return 0.5 * input * (1.0 + ivy.tanh(ivy.sqrt(2.0 / ivy.pi) * (input + 0.044715 * ivy.pow(input, 3.0))))


class GELUActivation(ivy.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    ivy.tanh(ivy.sqrt(2 / ivy.pi) * (x + 0.044715 * ivy.pow(x, 3)))) This is now written in C in ivy.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = ivy.gelu

    def _gelu_python(self, input: ivy.Array) -> ivy.Array:
        return input * 0.5 * (1.0 + ivy.erf(input / ivy.sqrt(2.0)))

    def forward(self, input: ivy.Array) -> ivy.Array:
        return self.act(input)


class FastGELUActivation(ivy.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: ivy.Array) -> ivy.Array:
        return 0.5 * input * (1.0 + ivy.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(ivy.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: ivy.Array) -> ivy.Array:
        return input * ivy.sigmoid(1.702 * input)


class ClippedGELUActivation(ivy.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    ivy.tanh(ivy.sqrt(2 / ivy.pi) * (x + 0.044715 * ivy.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: ivy.Array) -> ivy.Array:
        return ivy.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(ivy.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = ivy.sqrt(2 / ivy.pi)

    def forward(self, input: ivy.Array) -> ivy.Array:
        return 0.5 * input * (1 + ivy.tanh(self.precomputed_constant * (input + 0.044715 * ivy.pow(input, 3))))


class SiLUActivation(ivy.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: ivy.Array) -> ivy.Array:
        return ivy.silu(input)



class LinearActivation(ivy.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: ivy.Array) -> ivy.Array:
        return input


class LaplaceActivation(ivy.Module):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(self, input, mu=0.707107, sigma=0.282095):
        input = (input - mu).div(sigma * ivy.sqrt(2.0))
        return 0.5 * (1.0 + ivy.erf(input))


class ReLUSquaredActivation(ivy.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        relu_applied = ivy.relu(input)
        squared = ivy.square(relu_applied)
        return squared


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": ivy.GELU,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_tanh": IvyGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "linear": LinearActivation,
    "mish": ivy.Mish,
    "quick_gelu": QuickGELUActivation,
    "relu": ivy.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": ivy.ReLU6,
    "sigmoid": ivy.Sigmoid,
    "silu": ivy.SiLU,
    "swish": SiLUActivation,
    "tanh": ivy.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
relu = get_activation("relu")
relu6 = get_activation("relu6")
sigmoid = get_activation("sigmoid")
tanh = get_activation("tanh")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")