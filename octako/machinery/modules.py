import typing
import torch
import torch.nn as nn


class Lambda(nn.Module):

    def __init__(self, lambda_fn: typing.Callable[[], torch.Tensor]):

        super().__init__()
        self._lambda_fn = lambda_fn
    
    def forward(self, *x):

        return self._lambda_fn(*x)


class View(nn.Module):

    def __init__(self, sz: torch.Size):
        super().__init__()
        self._sz = sz

    def forward(self, x: torch.Tensor):
        return x.view(self._sz)


class BatchFlatten(nn.Module):

    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)

