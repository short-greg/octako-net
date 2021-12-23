from torch._C import Value
import torch.nn as nn
import torch


class Diverge(nn.Module):

    def __init__(self, mods):

        super().__init__()
        self._mods = mods
    
    def forward(self, *x: torch.Tensor):

        if len(x) != len(self._mods):
            raise ValueError(f"Number of inputs {len(x)} must equal number of modules {len(self._mods)}")

        return [mod(x_i) for mod, x_i in zip(self._mods, x)]


class Parallel(nn.Module):

    def __init__(self, mods):

        super().__init__()
        self._mods = mods
    
    def forward(self, x: torch.Tensor):

        if len(x) != len(self._mods):
            raise ValueError(f"Number of inputs {len(x)} must equal number of modules {len(self._mods)}")
        return [mod(x) for mod in self._mods]
