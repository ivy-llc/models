# global
import ivy


class PreNorm(ivy.Module):
    def __init__(
        self, dim, fn, key_dim=None, value_dim=None, eps=1e-05, device=None, v=None
    ):
        self._attention = fn
        self._norm = ivy.LayerNorm([dim], eps=eps, device=device)
        self._norm_key = (
            ivy.LayerNorm([key_dim], eps=eps, device=device)
            if ivy.exists(key_dim)
            else None
        )
        self._norm_value = (
            ivy.LayerNorm([value_dim], eps=eps, device=device)
            if ivy.exists(value_dim)
            else None
        )
        ivy.Module.__init__(self, v=v, device=device)

    def _forward(self, *args, **kwargs):
        args = list(args)
        args[0] = self._norm(args[0])
        if ivy.exists(self._norm_key):
            args[1] = self._norm_key(args[1])
        if ivy.exists(self._norm_value):
            args[2] = self._norm_value(args[2])
        return self._attention(*args, **kwargs)


class FeedForward(ivy.Module):
    def __init__(self, dim, dropout=0.0, device=None, v=None):
        self._net = ivy.Sequential(
            ivy.Linear(dim, dim, device=device),
            ivy.GELU(),
            ivy.Linear(dim, dim, device=device),
            ivy.Dropout(dropout),
            device=device,
        )
        ivy.Module.__init__(self, v=v)

    def _forward(self, x):
        return self._net(x)


def _perceiver_jax_weights_mapping(old_key, new_key):
    new_mapping = new_key
    if "proj_weights" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "a b -> b a"}
    elif "output/w" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "a b -> b a"}
    return new_mapping
