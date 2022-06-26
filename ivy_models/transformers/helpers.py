# global
import ivy


class PreNorm(ivy.Module):
    def __init__(self, dim, fn, context_dim=None, epsilon=None, dev_str=None, v=None):
        self._fn = fn
        self._norm = ivy.LayerNorm([dim], epsilon=epsilon, device=dev_str)
        if isinstance(context_dim, int):
            context_dim = [context_dim]
        self._norm_context = ivy.LayerNorm(context_dim, epsilon=epsilon, device=dev_str) if \
            ivy.exists(context_dim) else None
        ivy.Module.__init__(self, v=v, device=dev_str)

    def _forward(self, x, **kwargs):
        x = self._norm(x)
        if ivy.exists(self._norm_context):
            kwargs.update(context=self._norm_context(kwargs['context']))
        return self._fn(x, **kwargs)


class FeedForward(ivy.Module):
    def __init__(self, dim, dropout=0., dev_str=None, v=None):
        self._net = ivy.Sequential(
            ivy.Linear(dim, dim, device=dev_str),
            ivy.GELU(),
            ivy.Linear(dim, dim, device=dev_str),
            ivy.Dropout(dropout))
        ivy.Module.__init__(self, v=v)

    def _forward(self, x):
        return self._net(x)
