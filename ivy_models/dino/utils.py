import ivy

def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1. + ivy.erf(x/ivy.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        ivy.warn("mean is more than 2 std from [a, b] in trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    ivy.stop_gradient(tensor)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    ivy.random_uniform(low = 2 * l - 1, high = 2 * u - 1, out = tensor)
    # TODO: ivy.erfinv
    tensor = ivy.multiply(tensor, std * ivy.sqrt(2.))
    tensor = ivy.add(tensor, mean)
    tensor = ivy.clip(tensor, a, b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


if __name__ == "__main__":
    x = ivy.array([1., 2., 3., 5.])
    x = ivy.randint(-100, 100, shape = (10,5))
    truncated_tensor = trunc_normal_(x, std = .02)
    assert truncated_tensor.shape == x.shape
