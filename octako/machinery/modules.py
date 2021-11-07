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


class ListAdapter(nn.Module):

    def __init__(self, module: nn.Module):

        self._module = module

    def forward(self, **inputs):

        return self._module.forward(inputs)


class Reorder(nn.Module):
    """Reorder the inputs when they are a list
    """
    
    def __init__(self, input_map: typing.List[int]):
        """
        Args:
            input_map (typing.List[int]): 
        """
        assert len(input_map) == len(set(input_map))
        assert max(input_map) == len(input_map) - 1
        assert min(input_map) == 0

        self._input_map = input_map
    
    def forward(self, *inputs):
        result = [None] * len(self._input_map)

        for i, v in enumerate(inputs):
            result[self._input_map[i]] = v
        
        return result


class Selector(nn.Module):
    """Select a subset of the inputs passed in.
    """

    def __init__(self, input_count: int, to_select: typing.List[int]):
        """
        Args:
            input_count (int): The number of inputs past in
            to_select (typing.List[int]): The inputs to select
        """
        assert max(to_select) <= input_count - 1
        assert min(to_select) >= 0

        self._input_count = input_count
        self._to_select = to_select

    def forward(self, *inputs):

        assert len(inputs) == self._input_count

        result = []
        for i in self._to_select:
            result.append(inputs[i])
        return result
