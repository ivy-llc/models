# global
import ivy


class PreNorm(ivy.Module):
<<<<<<< HEAD
    def __init__(
        self, dim, fn, key_dim=None, value_dim=None, eps=1e-05, device=None, v=None
    ):
        self._attention = fn
=======
    def __init__(self, dim, fn, context_dim=None, eps=1e-05, device=None, v=None):
        self._fn = fn
>>>>>>> 1d929d5 (back to init)
        self._norm = ivy.LayerNorm([dim], eps=eps, device=device)
        if isinstance(context_dim, int):
            context_dim = [context_dim]
        self._norm_context = (
            ivy.LayerNorm(context_dim, eps=eps, device=device)
            if ivy.exists(context_dim)
            else None
        )
        ivy.Module.__init__(self, v=v, device=device)

<<<<<<< HEAD
    def _forward(self, *args, **kwargs):
        args = list(args)
        args[0] = self._norm(args[0])
        if ivy.exists(self._norm_key):
            args[1] = self._norm_key(args[1])
        if ivy.exists(self._norm_value):
            args[2] = self._norm_value(args[2])
        return self._attention(*args, **kwargs)
=======
    def _forward(self, x, **kwargs):
        x = self._norm(x)
        if ivy.exists(self._norm_context):
            kwargs.update(context=self._norm_context(kwargs["context"]))
        return self._fn(x, **kwargs)
>>>>>>> 1d929d5 (back to init)


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
