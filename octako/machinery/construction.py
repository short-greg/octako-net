from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from enum import Enum
import itertools
from os import stat
from torch._C import Module, float32
from torch.functional import norm
from octako.modules.activations import NullActivation, Scaler
from torch import nn
import torch
from octako.modules import objectives
from .networks import Network, Operation
import typing
from . import utils
from octako.modules import utils as util_modules
from functools import partial

"""
Overview: Modules for building layers in a network

They are meant to be extensible so that a builder can be replaced as long as it 
provides the same interface. 

They provide explicit functions like "relu" and also non-explicit functions
like "activation". The user can set activation to be any type of activation
as long as the interface is correct.
"""


class _Undefined:
    """
    use to have fields undefined in a dataclass
    since this is not resolved until Python 3.10 
    """
    def __eq__(self, other):
        return isinstance(self, other)
_UNDEFINED = _Undefined
UNDEFINED = partial(field, default_factory=_Undefined)


class OpFactory(ABC):
    
    @abstractmethod
    def produce(self, in_size: torch.Size) -> Operation:
        pass


class OpReversibleFactory(OpFactory):
    
    @abstractmethod
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        pass


class OpBuilder(ABC):


    @abstractproperty
    def product(self) -> Operation:
        pass


class NetBuilder(ABC):

    @abstractproperty
    def product(self) -> Network:
        pass


@dataclass
class ActivationFactory(OpReversibleFactory):

    kwargs: dict = field(default_factory=dict)
    torch_act: typing.Type[nn.Module] = nn.ReLU

    def produce(self, in_size: torch.Size) -> Operation:
        
        return Operation(self.torch_act(**self.kwargs), in_size)
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return self.produce(in_size)


@dataclass
class NormalizerFactory(OpReversibleFactory):

    eps: float=1e-4
    momentum: float=1e-1
    track_running_stats: bool=True
    affine: bool=True
    device: str="cpu"
    dtype: torch.dtype= torch.float32
    torch_normalizer: typing.Type[nn.Module] = nn.BatchNorm1d

    def produce(self, in_size: torch.Size) -> Operation:
        return Operation(self.torch_normalizer(
            in_size[1], eps=self.eps, momentum=self.momentum, 
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            device=self.device, dtype=self.dtype
        ), in_size)
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return self.produce(in_size)


@dataclass
class LinearFactory(OpReversibleFactory):

    out_features: int
    bias: bool=True
    device: str="cpu"
    dtype: torch.dtype= torch.float32

    def produce(self, in_size: torch.Size) -> Operation:
        return Operation(
            nn.Linear(
                in_size[1], self.out_features, bias=self.bias, 
                device=self.device, dtype=self.dtype
            ), 
            torch.Size(in_size[0], self.out_features)
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return Operation(
            nn.Linear(
                self.out_features, in_size[1], bias=self.bias, 
                device=self.device, dtype=self.dtype
            ), 
            in_size
        )


@dataclass
class DropoutFactory(OpReversibleFactory):

    p: float=0.2
    inplace: bool=False
    dropout: typing.Type[nn.Module] = nn.Dropout

    def produce(self, in_size: torch.Size) -> Operation:
        return Operation(
            self.dropout(p=self.p, inplace=self.inplace),
            torch.Size(in_size)
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return self.produce(in_size)


@dataclass
class DimAggregateFactory(OpFactory):

    dim: int=1
    index: int=None
    torch_agg: typing.Callable[[int], torch.Tensor]=torch.mean

    def produce(self, in_size: torch.Size) -> Operation:
        f = lambda x: (
            self.torch_agg(x, dim=self.dim) if self.index is None
            else self.torch_agg(x, dim=self.dim)[self.index]
        )
        out_size = in_size[:self.dim] + in_size[self.dim + 1:]
        return Operation(
            util_modules.Lambda(f), out_size
        )


@dataclass
class ConvolutionFactory(OpReversibleFactory):

    out_features: int
    k: typing.Union[int, typing.Tuple]=1
    stride: typing.Union[int, typing.Tuple]=1
    padding: typing.Union[int, typing.Tuple]=0
    torch_conv: typing.Type[nn.Module]= nn.Conv2d
    torch_deconv: typing.Type[nn.Module]= nn.ConvTranspose2d
    kwargs: dict=field(default_factory=dict)

    def produce(self, in_size: torch.Size):

        out_sizes = utils.calc_conv_out(in_size, self.k, self.stride, self.padding)
        out_size = torch.Size([-1, self.out_features, *out_sizes])
        return Operation(
            self.torch_conv(in_size[1], self.out_features, 
            self.k, self.stride, padding=self.padding, **self.kwargs),
            out_size
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:        
        out_sizes = utils.calc_conv_transpose_out(in_size, self.k, self.stride, self.padding)
        out_size = torch.Size([-1, self.out_features, *out_sizes])
        return Operation(
            self.torch_conv(in_size[1], self.out_features, 
            self.k, self.stride, padding=self.padding, **self.kwargs),
            out_size
        )


@dataclass
class PoolFactory(OpReversibleFactory):

    k: typing.Union[int, typing.Tuple]=1
    stride: typing.Union[int, typing.Tuple]=1
    padding: typing.Union[int, typing.Tuple]=0
    torch_pool: typing.Type[nn.Module]= nn.MaxPool2d
    torch_unpool: typing.Type[nn.Module]= nn.MaxUnpool2d
    kwargs: dict=field(default_factory=dict)

    def produce(self, in_size: torch.Size):
        
        out_sizes = utils.calc_max_pool_out(in_size, self.k, self.stride, self.padding)
        out_size = torch.Size([-1, in_size[1], *out_sizes])
        return Operation(
            self.torch_pool(in_size[1], in_size[1], self.k, self.stride, padding=self.padding),
            out_size
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        out_sizes = utils.calc_maxunpool_out(in_size, self.k, self.stride, self.padding)
        out_size = torch.Size([-1, in_size[1], *out_sizes])
        return Operation(
            self.torch_unpool(in_size[1], in_size[1], self.k, self.stride, padding=self.padding),
            out_size
        )
    


@dataclass
class ViewFactory(OpReversibleFactory):

    view: torch.Size
    keepbatch: bool=True

    def produce(self, in_size: torch.Size):

        out_size = in_size[0:1] + self.view if self.keepbatch else self.view
            
        return Operation(
            util_modules.View(self.view, keepbatch=self.keepbatch),
            out_size
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return Operation(
            util_modules.View(in_size, keepbatch=self.keepbatch),
            in_size
        )


@dataclass
class RepeatFactory(OpFactory):

    repeat_by: typing.List[int]
    keepbatch: bool=True

    def produce(self, in_size: torch.Size):
        out_size = []
        for x, y in zip(in_size[1:], self.repeat_by):
            if x == -1:
                out_size.append(-1)
            else:
                out_size.append(x * y)

        return Operation(
            util_modules.Lambda(lambda x: x.repeat(1, *self.repeat_by)), 
            torch.Size([-1, *out_size])
        )


@dataclass
class NullFactory(OpReversibleFactory):
    
    def produce(self, in_size: torch.Size):

        return Operation(
            NullActivation(), in_size
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return Operation(
            NullActivation(), in_size
        )


@dataclass
class ScalerFactory(OpFactory):
    
    def produce(self, in_size: torch.Size):

        return Operation(
            Scaler(), in_size
        )


@dataclass
class TorchLossFactory(OpFactory):

    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    torch_loss_cls: typing.Type[nn.Module]= nn.MSELoss
    
    def produce(self, in_size: torch.Size):

        return Operation(
            self.torch_loss_cls(reduction=self.reduction_cls.as_str()),
            self.reduction_cls.get_out_size(in_size)
        )


@dataclass
class RegularizerFactory(OpFactory):

    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    regularizer_cls: typing.Type[objectives.Regularizer]= objectives.L2Reg
    
    def produce(self, in_size: torch.Size):

        return Operation(
            self.regularizer_cls(reduction=self.reduction_cls()),
            self.reduction_cls.get_out_size(in_size)
        )


@dataclass
class LossFactory(OpFactory):

    loss_cls: typing.Type[objectives.Loss]
    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    
    def produce(self, in_size: torch.Size):

        return Operation(
            self.loss_cls(reduction=self.reduction_cls()),
            self.reduction_cls.get_out_size(in_size)
        )


@dataclass
class ValidationFactory(OpFactory):

    validation_cls: typing.Type[objectives.Loss]=objectives.ClassificationFitness
    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    
    def produce(self, in_size: torch.Size):

        return Operation(
            self.validation_cls(reduction=self.reduction_cls()),
            self.reduction_cls.get_out_size(in_size)
        )


@dataclass
class AggregateFactory(OpFactory):

    aggregator_fn: typing.Callable[[torch.Tensor], torch.Tensor]=torch.mean
    weights: typing.List=None

    def produce(self, in_size: torch.Size):

        def aggregator(*x):
            if self.weights is not None:
                x = [
                    el * w for el, w in zip(x, self.weights)
                ]
            
            return self.aggregator_fn(
                *x
            )

        return Operation(
            util_modules.Lambda(aggregator),
            in_size
        )
