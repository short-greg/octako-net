"""
Network modules 
"""

import torch.nn as nn
import torch
import typing


class NullActivation(nn.Module):

    def forward(self, x):
        return x


class Scaler(nn.Module):
    """Scale the input between a lower bound and upper bound
    """

    def __init__(
        self, lower: float=0., 
        upper: float=1., 
        n_update_limit=None,
    ):
        """initializer

        Args:
            lower (float, optional): Lower bound on the output. Defaults to 0..
            upper (float, optional): Upper bound on the output. Defaults to 1..
            n_update_limit (_type_, optional): Number of times to update
              the scaling parameters. Defaults to None.
        """
        super().__init__()
        assert lower < upper
        self._lower = lower
        self._upper = upper
        self._n_update_limit = n_update_limit
        self._n_updates = 0
        self._lower_vec = None
        self._upper_vec = None
        self._vecs_set = False
    
    def _update_scaling_parameters(self, x):
        """
        Args:
            x (_type_): input into the module
        """
        if self._vecs_set is False:
            self._lower_vec = torch.min(x, dim=0, keepdim=True)[0]
            self._upper_vec = torch.max(x, dim=0, keepdim=True)[0]
            self._vecs_set = True
        else:
            self._lower_vec = torch.min(
                torch.min(x, dim=0, keepdim=True)[0],
                self._lower_vec
            )
            self._upper_vec = torch.max(
                torch.max(x, dim=0, keepdim=True)[0],
                self._lower_vec
            )
    
    def forward(self, x):
        """Scale the input

        Args:
            x (torch.Tensor): input to the module

        Returns:
            torch.Tensor: Scaled input
        """

        to_update_parameters = self.training and (
            self._n_update_limit is None 
            or self._n_updates < self._n_update_limit
        )
        if to_update_parameters:
            self._update_scaling_parameters(x)
        elif not self.training and not self._vecs_set:
            return x

        # scale to between 0 and 1. add 1e-5 to ensure there is no / 0
        scaled_x = (x - self._lower_vec) / (self._upper_vec - self._lower_vec + 1e-5) 
        
        if self._lower != 0. or self._upper != 1.:
            return scaled_x * (self._upper - self._lower) + self._lower

        return scaled_x


class Diverge(nn.Module):
    """
    Pass each input onto a separate module
    """

    def __init__(self, mods: typing.List[nn.Module]):
        """initializer
        Args:
            mods (list[nn.Module]): Modules to process each input 
        """

        super().__init__()
        self._mods = mods
    
    def forward(self, *x: torch.Tensor):
        """Pass each input through the submodules

        Raises:
            ValueError: Number of inputs does not match the number of modules

        Returns:
            list[torch.Tensor] 
        """

        if len(x) != len(self._mods):
            raise ValueError(f"Number of inputs {len(x)} must equal number of modules {len(self._mods)}")

        return [mod(x_i) for mod, x_i in zip(self._mods, x)]


class Multi(nn.Module):

    def __init__(self, mods: typing.List[nn.Module]):
        """initializer

        Args:
            mods (typing.List[nn.Module]): Modules for processing the input
        """

        super().__init__()
        self._mods = mods
    
    def forward(self, x: torch.Tensor):
        """Pass input through all submodules

        Raises:
            ValueError: Number of inputs does not match the number of modules

        Returns:
            list[torch.Tensor] 
        """

        if len(x) != len(self._mods):
            raise ValueError(f"Number of inputs {len(x)} must equal number of modules {len(self._mods)}")
        return [mod(x) for mod in self._mods]


class Lambda(nn.Module):
    """
    Define a module inline
    """

    def __init__(self, lambda_fn: typing.Callable[[], torch.Tensor]):
        """initializer

        Args:
            lambda_fn (typing.Callable[[], torch.Tensor]): Function to process the tensor
        """

        super().__init__()
        self._lambda_fn = lambda_fn
    
    def forward(self, *x: torch.Tensor):
        """Execute the lambda function

        Returns:
            list[torch.Tensor] or torch.Tensor 
        """

        return self._lambda_fn(*x)


class View(nn.Module):
    """Reshape the input 
    """

    def __init__(self, *size):
        """initializer

        Args:
            size (torch.Size): Size to reshape to 
        """
        super().__init__()
        self._sz = size

    def forward(self, x: torch.Tensor):
        """

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: Reshaped tensor
        """
        return x.view(*self._sz)


class Flatten(nn.Module):
    """Flatten the tensor
    """

    def __init__(self, keepbatch: bool=False):
        """_summary_

        Args:
            keepbatch (bool, optional): Whether to flatten the batch dim or keep the batch dim. 
            Defaults to False.
        """
        super().__init__()
        self._keepbatch = keepbatch

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): input

        Returns:
            torch.Tensor: Flattened tensor
        """
        if self._keepbatch:
            return x.view(x.size(0), -1)
        else:
            return x.view(-1)


class ListAdapter(nn.Module):
    """Adapts a module that takes in a list as input to the forward method
    """

    def __init__(self, module: nn.Module):
        """initializer

        Args:
            module (nn.Module): module to adapt to
        """

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
        """Select inputs to output

        Returns:
            list[torch.Tensor]: Selected inputs
        """

        assert len(inputs) == self._input_count

        result = []
        for i in self._to_select:
            result.append(inputs[i])
        return result


class Concat(nn.Module):
    """Concatenate the inputs along a dimension
    """

    def __init__(self, dim: int=1):
        super().__init__()
        self._dim = dim
    
    def forward(self, *x: torch.Tensor):
        return torch.cat(x, dim=self._dim)


class Stack(nn.Module):
    
    def forward(self, *x: torch.Tensor):
        return torch.stack(x)


class Printf(nn.Module):
    """print an intermediate value in a network 
    """

    def __init__(self, f, print_f=print):
        """initializer

        Args:
            f (function): evaluate what to print
            print_f (_type_, optional): function to output the value. Defaults to print.
        """
        super().__init__()
        self._f = f
        self._print_f = print_f

    def forward(self, x):
        self._print_f(self._f(x))
        return x
