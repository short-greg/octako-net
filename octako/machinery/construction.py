from abc import ABC, abstractmethod, abstractproperty
from dataclasses import InitVar, asdict, dataclass, field
from enum import Enum
import itertools
from os import stat
from torch._C import Module, float32
from torch.functional import norm
from octako.modules.activations import NullActivation, Scaler
from torch import nn
import torch
from octako.modules import objectives
from .networks import Network, NetworkConstructor, Operation, Port
import typing
from . import utils
from octako.modules import utils as util_modules
from functools import partial, singledispatchmethod

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


@dataclass
class BaseInput(object):

    size: torch.Size
    name: str="x"


@dataclass
class BaseNetwork(object):
    """
    Base network to build off of. Can be used in the build() method
    """

    constructor: NetworkConstructor
    ports: typing.List[Port]

    def __post_init__(self):

        if isinstance(self.ports, Port):
            self.ports = [self.ports]

    @singledispatchmethod
    def update(self, ports):
        self.ports = ports

    @update.register
    def update(self, ports: Port):
        self.ports = [ports]

    @classmethod
    def from_base_input(cls, base: BaseInput):
        network_constructor = NetworkConstructor(Network())
        in_size = base.size
        input_name = base.name
        port = network_constructor.add_tensor_input(
            input_name, in_size
        )
        return BaseNetwork(network_constructor, port)

    @classmethod
    def from_base_inputs(cls, base: typing.List[BaseInput]):
        network_constructor = NetworkConstructor(Network())
        ports = []
        for x in base:
            in_size = x.size
            input_name = x.name
            ports.append(network_constructor.add_tensor_input(
                input_name, in_size
            )[0])
            
        return BaseNetwork(network_constructor, ports)


@dataclass
class AbstractConstructor(ABC):

    def __post_init__(self):
        
        self._base_data = asdict(self)
    
    def reset(self, reset_base_data: bool=True):

        if reset_base_data:
            for k, v in self._base_data.items():
                self.__setattr__(k, v)


class OpFactory(AbstractConstructor):
    
    @abstractmethod
    def produce(self, in_sizes: typing.List[torch.Size]) -> Operation:
        pass


class OpReversibleFactory(AbstractConstructor):
    
    @abstractmethod
    def produce(self, in_sizes: typing.List[torch.Size]) -> Operation:
        pass

    @abstractmethod
    def produce_reverse(self, in_sizes: typing.List[torch.Size]) -> Operation:
        pass


class OpBuilder(AbstractConstructor):

    @abstractproperty
    def product(self) -> Operation:
        pass


class NetBuilder(AbstractConstructor):

    @abstractproperty
    def product(self) -> Network:
        pass


@dataclass
class ActivationFactory(OpReversibleFactory):

    torch_act_cls: typing.Type[nn.Module] = nn.ReLU
    kwargs: dict = field(default_factory=dict)

    def produce(self, in_sizes: typing.List[torch.Size]) -> Operation:
        
        return Operation(self.torch_act_cls(**self.kwargs), in_sizes[0])
    
    def produce_reverse(self, in_sizes: typing.List[torch.Size]) -> Operation:
        return self.produce(in_sizes)


@dataclass
class NormalizerFactory(OpReversibleFactory):

    eps: float=1e-4
    momentum: float=1e-1
    track_running_stats: bool=True
    affine: bool=True
    device: str="cpu"
    dtype: torch.dtype= torch.float32
    torch_normalizer_cls: typing.Type[nn.Module] = nn.BatchNorm1d

    def produce(self, in_sizes: typing.List[torch.Size]) -> Operation:
        return Operation(self.torch_normalizer_cls(
            in_sizes[0][1], eps=self.eps, momentum=self.momentum, 
            affine=self.affine,
            track_running_stats=self.track_running_stats,
            device=self.device, dtype=self.dtype
        ), in_sizes[0])
    
    def produce_reverse(self, in_sizes: typing.List[torch.Size]) -> Operation:
        return self.produce(in_sizes)


@dataclass
class LinearFactory(OpReversibleFactory):

    out_features: int
    bias: bool=True
    device: str="cpu"
    dtype: torch.dtype= torch.float32

    def produce(self, in_sizes: typing.List[torch.Size]) -> Operation:
        return Operation(
            nn.Linear(
                in_sizes[0][1], self.out_features, bias=self.bias, 
                device=self.device, dtype=self.dtype
            ), 
            torch.Size(in_sizes[0][0], self.out_features)
        )
    
    def produce_reverse(self, in_sizes: typing.List[torch.Size]) -> Operation:
        return Operation(
            nn.Linear(
                self.out_features, in_sizes[0][1], bias=self.bias, 
                device=self.device, dtype=self.dtype
            ), 
            in_sizes[0]
        )


@dataclass
class DropoutFactory(OpReversibleFactory):

    p: float=0.2
    inplace: bool=False
    dropout_cls: typing.Type[nn.Module] = nn.Dropout

    def produce(self, in_sizes: typing.List[torch.Size]) -> Operation:
        return Operation(
            self.dropout_cls(p=self.p, inplace=self.inplace),
            torch.Size(in_sizes[0])
        )
    
    def produce_reverse(self, in_sizes: typing.List[torch.Size]) -> Operation:
        return self.produce(in_sizes)


@dataclass
class DimAggregateFactory(OpFactory):

    dim: int=1
    index: int=None
    torch_agg_fn: typing.Callable[[int], torch.Tensor]=torch.mean

    def produce(self, in_sizes: typing.List[torch.Size]) -> Operation:
        f = lambda x: (
            self.torch_agg_fn(x, dim=self.dim) if self.index is None
            else self.torch_agg_fn(x, dim=self.dim)[self.index]
        )
        out_size = in_sizes[0][:self.dim] + in_sizes[0][self.dim + 1:]
        return Operation(
            util_modules.Lambda(f), out_size
        )


@dataclass
class ConvolutionFactory(OpReversibleFactory):

    out_features: int
    k: typing.Union[int, typing.Tuple]=1
    stride: typing.Union[int, typing.Tuple]=1
    padding: typing.Union[int, typing.Tuple]=0
    torch_conv_cls: typing.Type[nn.Module]= nn.Conv2d
    torch_deconv_cls: typing.Type[nn.Module]= nn.ConvTranspose2d
    kwargs: dict=field(default_factory=dict)

    def produce(self, in_sizes: typing.List[torch.Size]):

        out_sizes = utils.calc_conv_out(in_sizes[0], self.k, self.stride, self.padding)
        out_size = torch.Size([-1, self.out_features, *out_sizes])
        return Operation(
            self.torch_conv_cls(in_sizes[0][1], self.out_features, 
            self.k, self.stride, padding=self.padding, **self.kwargs),
            out_size
        )
    
    def produce_reverse(self, in_sizes: typing.List[torch.Size]) -> Operation:        
        out_sizes = utils.calc_conv_transpose_out(in_sizes[0], self.k, self.stride, self.padding)
        out_size = torch.Size([-1, self.out_features, *out_sizes])
        return Operation(
            self.torch_deconv_cls(in_sizes[0][1], self.out_features, 
            self.k, self.stride, padding=self.padding, **self.kwargs),
            out_size
        )


@dataclass
class PoolFactory(OpReversibleFactory):

    k: typing.Union[int, typing.Tuple]=1
    stride: typing.Union[int, typing.Tuple]=1
    padding: typing.Union[int, typing.Tuple]=0
    torch_pool_cls: typing.Type[nn.Module]= nn.MaxPool2d
    torch_unpool_cls: typing.Type[nn.Module]= nn.MaxUnpool2d
    kwargs: dict=field(default_factory=dict)

    def produce(self, in_sizes: typing.List[torch.Size]):
        
        out_sizes = utils.calc_max_pool_out(in_sizes[0], self.k, self.stride, self.padding)
        out_size = torch.Size([-1, in_sizes[0][1], *out_sizes])
        return Operation(
            self.torch_pool_cls(in_sizes[0][1], in_sizes[0][1], self.k, self.stride, padding=self.padding),
            out_size
        )
    
    def produce_reverse(self, in_sizes: typing.List[torch.Size]) -> Operation:
        out_sizes = utils.calc_maxunpool_out(in_sizes[0], self.k, self.stride, self.padding)
        out_size = torch.Size([-1, in_sizes[0][1], *out_sizes])
        return Operation(
            self.torch_unpool_cls(in_sizes[0][1], in_sizes[0][1], self.k, self.stride, padding=self.padding),
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
    
    def produce(self, in_sizes: typing.List[torch.Size]):

        return Operation(
            NullActivation(), in_sizes[0]
        )
    
    def produce_reverse(self, in_sizes: typing.List[torch.Size]) -> Operation:
        return Operation(
            NullActivation(), in_sizes[0]
        )


@dataclass
class ScalerFactory(OpFactory):
    
    def produce(self, in_sizes: typing.List[torch.Size]):

        return Operation(
            Scaler(), in_sizes[0]
        )


@dataclass
class TorchLossFactory(OpFactory):

    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    torch_loss_cls: typing.Type[nn.Module]= nn.MSELoss
    
    def produce(self, in_sizes: typing.List[torch.Size]):

        return Operation(
            self.torch_loss_cls(reduction=self.reduction_cls.as_str()),
            self.reduction_cls.get_out_size(in_sizes[0])
        )


@dataclass
class RegularizerFactory(OpFactory):

    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    regularizer_cls: typing.Type[objectives.Regularizer]= objectives.L2Reg
    
    def produce(self, in_sizes: typing.List[torch.Size]):

        return Operation(
            self.regularizer_cls(reduction=self.reduction_cls()),
            self.reduction_cls.get_out_size(in_sizes[0])
        )


@dataclass
class LossFactory(OpFactory):

    loss_cls: typing.Type[objectives.Loss]
    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    
    def produce(self, in_sizes: typing.List[torch.Size]):

        return Operation(
            self.loss_cls(reduction=self.reduction_cls()),
            self.reduction_cls.get_out_size(in_sizes[0])
        )


@dataclass
class ValidationFactory(OpFactory):

    validation_cls: typing.Type[objectives.Loss]=objectives.ClassificationFitness
    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    
    def produce(self, in_sizes: typing.List[torch.Size]):

        return Operation(
            self.validation_cls(reduction=self.reduction_cls()),
            self.reduction_cls.get_out_size(in_sizes[0])
        )


@dataclass
class AggregateFactory(OpFactory):

    aggregator_fn: typing.Callable[[torch.Tensor], torch.Tensor]=torch.mean
    weights: typing.List=None

    def produce(self, in_sizes: typing.List[torch.Size]):

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
            in_sizes[0]
        )


@dataclass
class LinearLayerBuilder(OpBuilder):

    in_size: torch.Size
    activation: ActivationFactory=ActivationFactory(torch.nn.ReLU)
    dropout: DropoutFactory=DropoutFactory()
    normalizer: NormalizerFactory=NormalizerFactory()
    bias: bool=True

    def __post_init__(self):
        super().__post_init__()
        self._cur_in_size = self.in_size
        self._product: nn.Sequential = nn.Sequential()
        if self.activation is None:
            self.activation = NullFactory()
        if self.normalizer is None:
            self.normalizer = NullFactory()
        if self.dropout is None:
            self.dropout = NullFactory()
    
    def _add_op(self, op: Operation):
        self._cur_in_size = op.out_size
        self._product.add_module(op.op)

    def build_activation(self):
        self._add_op(self.activation.produce(self._cur_in_size))
    
    def build_linear(self, out_features):
        self._add_op(
            LinearFactory(out_features, bias=self.bias).produce(self._cur_in_size)
        )

    def build_normalizer(self):
        self._add_op(
            self.normalizer.produce(self._cur_in_size)
        )

    def build_dropout(self):
        self._add_op(
            self.dropout.produce(self._cur_in_size)
        )

    @property
    def product(self) -> Operation:
        return Operation(self._product, self._cur_in_size)
    
    def reset(self, in_size: torch.Size=None, reset_base_data: bool=True) -> None:

        super().reset(reset_base_data)
        self._product = nn.Sequential()
        self._cur_in_size = self.in_size
        self.in_size = utils.coalesce(in_size, self.in_size)
    
    def build_standard(self, out_features: int):
        self.build_dropout()
        self.build_linear(out_features)
        self.build_normalizer()
        self.build_activation()


@dataclass
class FeedForwardBuilder(NetBuilder):

    # pass in the base of the network
    base: InitVar(typing.Union[BaseInput, BaseNetwork])
    out_features: typing.List[int]
    base_name: str="layer"
    labels: typing.List[str]=field(default_factory=partial(list, "linear"))
    activation: ActivationFactory=ActivationFactory(torch_act=nn.ReLU)
    normalizer: NormalizerFactory=None
    dropout: DropoutFactory=None

    def __post_init__(self, base: typing.Union[BaseInput, BaseNetwork]):
        
        super().__post_init__()

        self._cur_base = self._setup_base(base)
        self._layer_builder = LinearLayerBuilder(self._cur_base.ports[0].size)
        self._n_layers = 0

    @property
    def n_layers(self):
        pass

    def build_layer(self, layer_num: int, name: str=None, labels: typing.List[str]=None):
        
        if 0 < layer_num <= len(self.out_features):
            raise ValueError(f"{layer_num} must be in range [0, {len(self.out_features)})")

        self._layer_builder = LinearLayerBuilder(
            dropout=self.dropout,
            normalizer=self.normalizer,
            in_size=self._cur_base.ports[0].size,
            activation=self.activation
        )

        self._layer_builder.reset(in_size=self._cur_base.ports[0].size)
        name = utils.coalesce(name, f'{self.base_name}_{self._n_layers}')
        labels = utils.coalesce(labels, self.labels)
        if self.dropout: self._layer_builder.build_dropout()
        self._layer_builder.build_linear(self.out_features[layer_num])
        if self.normalizer: self._layer_builder.build_normalizer()
        if self.activation: self._layer_builder.build_activation()
        
        op = self._layer_builder.product
        self._cur_base.update(self._cur_base.constructor.add_op(
            name, op, self._port, labels
        ))
        self._n_layers += 1

    @singledispatchmethod
    def _setup_base(self, base) -> BaseNetwork:
        return base

    @_setup_base.register
    def _(self, base: BaseInput) -> BaseNetwork:

        return BaseNetwork.from_base_input(base)

    def reset(
        self, base: typing.Union[BaseInput, BaseNetwork]=None,
        reset_base_data: bool=True
    ):
        super().reset(reset_base_data)
        if base is not None:
            self._cur_base = self._setup_base(base)
            self._n_layers = 0

    def product(self) -> Network:
        return self._cur_base.constructor.net
