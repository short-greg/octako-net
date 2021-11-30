"""
Overview: Modules for building layers in a network

They are meant to be extensible so that a builder can be replaced as long as it 
provides the same interface. 

They provide explicit functions like "relu" and also non-explicit functions
like "activation". The user can set activation to be any type of activation
as long as the interface is correct.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from os import stat
from torch.functional import norm
from octako.modules.activations import NullActivation, Scaler
from torch import nn
import torch
from octako.modules import objectives
from .networks import Link, Network, Node, Operation, Parameter, Port
import typing
from typing import Optional as Opt
from . import utils
from octako.modules import utils as util_modules
from .networks import In, OpNode, SubNetwork, InterfaceNode


class UNDEFINED:
    """
    use to have fields undefined in a dataclass
    since this is not resolved until Python 3.10 
    """
    pass


def is_undefined(val):
    return isinstance(val, UNDEFINED) or val == UNDEFINED



class NetworkBuilder(object):
    """
    Builder class with convenience methods for building networks
    - Handles adding nodes to the network to simplify that process
    - Do not need to specify the incoming ports if the incoming ports
    -  are the outputs of the previously added node
    - Can set base labels that will apply to all nodes added to the network
    -  that use the convenience methods (not add_node)
    """

    def __init__(self, network: Network=None):
        """
        network: Network - The network to construct
        """
        self._network = network or Network()
        self._subnets: typing.Dict[str, Network] = {}
        self.base_labels: typing.Optional[typing.List[str]] = None
        self._cur_ports: typing.List[Port] = None
    
    def add_subnets(self, **kwargs: typing.Dict[str, Network]): # name: str, network: Network):
        
        for name, network in kwargs.items():
            if name in self._subnets:
                raise ValueError(f"Subnet by name {name} already exists")
        
            self._subnets[name] = SubNetwork(name, network)
    
    def _coalesce(self, ports, labels):
        out_labels = []
        if self.base_labels is not None:
            out_labels += self.base_labels
        if labels is not None:
            out_labels += labels

        ports = utils.coalesce(ports, self._cur_ports)
        return ports, labels

    def _set_ports(self, ports):
        self._cur_ports = ports
        return ports

    @property
    def net(self):
        return self._network
    
    def sub(self, key: str):
        return self._subnets[key]
    
    def __getitem__(self, name: typing.Union[str, typing.List]) -> Node:
        return self._network[name]

    def add_node(self, node: Node):
        self._network.add_node(node)
        return self._set_ports(node.ports)

    def add_op(
        self, name: str, op: Operation, in_: typing.Union[Port, typing.List[Port]]=None, 
        labels: typing.List[typing.Union[typing.Iterable[str], str]]=None
    ) -> typing.List[Port]:
        """[summary]

        Args:
            name (str): The name of the node
            op (Operation): Operation for the node to perform
            in_ (typing.Union[Port, typing.List[Port]]): The ports feeding into the node
            labels (typing.List[str], optional): Labels for the node to be used for searching. Defaults to None. 

        Raises:
            KeyError: If the name for the node already exists in the network.

        Returns:
            typing.List[Port]: The ports feeding out of the node
        """

        if isinstance(in_, Port):
            in_ = [in_]
        
        in_, labels = self._coalesce(in_, labels)
        node = OpNode(name, op.op, in_, op.out_size, labels)
        return self._set_ports(self._network.add_node(node))
    
    def add_module_op(
        self, name: str, mod: nn.Module, 
        out_size: typing.Union[typing.List[torch.Size], torch.Size], 
        in_: typing.Union[Port, typing.List[Port]]=None, 
        labels: typing.List[typing.Union[typing.Iterable[str], str]]=None
    ): 
        in_, labels = self._coalesce(in_, labels)
        node = OpNode(name, mod, in_, out_size, labels)
        return self._set_ports(self._network.add_node(node))

    def add_lambda_op(
        self, name: str, f: typing.Callable[[], torch.Tensor], 
        out_size: typing.Union[typing.List[torch.Size], torch.Size], 
        in_: typing.Union[Port, typing.List[Port]], 
        labels: typing.List[typing.Union[typing.Iterable[str], str]]=None
    ): 
        in_, labels = self._coalesce(in_, labels)
        node = OpNode(name, util_modules.Lambda(f), in_, out_size, labels)
        return self._set_ports(self._network.add_node(node))

    def add_subnet_interface(self, name: str, subnet_name: str, in_links: typing.List[Link], out_ports: typing.List[Port]):
        subnet = self._subnets[subnet_name]
        node = InterfaceNode(name, subnet, out_ports, in_links)
        return self._set_ports(self._network.add_node(node))
    
    def add_input(self, name, sz: torch.Size, value_type: typing.Type, default_value, labels: typing.List[typing.Union[typing.Iterable[str], str]]=None, annotation: str=None):
        
        _, labels = self._coalesce(None, labels)
        node = In(name, sz, value_type, default_value, labels, annotation)
        return self._set_ports(self._network.add_node(node))

    def add_tensor_input(self, name, sz: torch.Size, default_value: torch.Tensor=None, labels: typing.List[typing.Union[typing.Iterable[str], str]]=None, annotation: str=None, device: str='cpu'):
        
        _, labels = self._coalesce(None, labels)
        node = In.from_tensor(name, sz, default_value, labels, annotation, device)
        return self._set_ports(self._network.add_node(node))

    def add_scalar_input(self, name, default_type: typing.Type, default_value, labels: typing.Set[str]=None, annotation: str=None):

        _, labels = self._coalesce(None, labels)
        node = In.from_scalar(name, default_type, default_value, labels, annotation)
        return self._set_ports(self._network.add_node(node))

    def add_parameter(
        self, name: str, sz: torch.Size, reset_func: typing.Callable[[torch.Size], torch.Tensor], 
        labels: typing.List[typing.Union[typing.Iterable[str], str]]=None, annotation: str=None
    ):
        _, labels = self._coalesce(None, labels)
        node = Parameter(name, sz, reset_func, labels, annotation)
        return self._set_ports(self._network.add_node(node))
    
    def set_default_interface(self, ins: typing.List[typing.Union[Port, str]], outs: typing.List[typing.Union[Port, str]]):
        self._network.set_default_interface(
            ins, outs
        )

    @classmethod
    def build_new(cls, inputs: typing.List[In]=None):

        return NetworkBuilder(Network(inputs))


@dataclass
class BaseInput(object):

    size: torch.Size
    name: str="x"


def get_undefined(dataclass_obj) -> typing.Optional[str]:
    for k, v in asdict(dataclass_obj).items():
        if is_undefined(v):
            return k
    return None


def check_undefined(f):

    def _(dataclass_obj, *args, **kwargs):
        k = get_undefined(dataclass_obj)
        if k is not None:
            raise ValueError(f"Parameter {k} has not been defined for {type(dataclass_obj)}")

        return f(dataclass_obj, *args, **kwargs)
    return _


@dataclass
class BaseNetwork(object):
    """
    Network to build off of containing ports and 
    a constructor.
    """

    constructor: NetworkBuilder
    ports: typing.List[Port]

    def __post_init__(self):

        if isinstance(self.ports, Port):
            self.ports = [self.ports]

    @classmethod
    def from_base_input(cls, base: BaseInput):
        network_constructor = NetworkBuilder()
        in_size = base.size
        input_name = base.name
        port = network_constructor.add_tensor_input(
            input_name, in_size
        )
        return BaseNetwork(network_constructor, port)

    @classmethod
    def from_base_inputs(cls, base: typing.List[BaseInput]):
        network_constructor = NetworkBuilder()
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

    def reset(self):        
        for k, v in self._base_data.items():
            if isinstance(v, AbstractConstructor):
               v.reset() 
            self.__setattr__(k, v)
    
    def is_undefined(self):
        return get_undefined(self) is not None


class OpFactory(AbstractConstructor):
    
    @abstractmethod
    def produce(self, *in_size: torch.Size) -> Operation:
        pass


class OpReversibleFactory(AbstractConstructor):
    
    @abstractmethod
    def produce(self, *in_size: torch.Size) -> Operation:
        pass

    @abstractmethod
    def produce_reverse(self, *in_size: torch.Size) -> Operation:
        pass


class OpDirector(AbstractConstructor):

    @abstractmethod
    def produce(self) -> Operation:
        pass


class NetDirector(AbstractConstructor):

    base: BaseInput

    @abstractmethod
    def produce(self) -> Network:
        pass

    @abstractmethod
    def append(self, base_network: BaseNetwork) -> Network:
        pass

@dataclass
class ActivationFactory(OpReversibleFactory):

    torch_act_cls: typing.Type[nn.Module] = nn.ReLU
    kwargs: dict = field(default_factory=dict)

    def produce(self, in_size: torch.Size) -> Operation:
        return Operation(self.torch_act_cls(**self.kwargs), in_size)
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return self.produce(in_size)


@dataclass
class NormalizerFactory(OpReversibleFactory):

    torch_normalizer_cls: typing.Type[nn.Module] = nn.BatchNorm1d
    eps: float=1e-4
    momentum: float=1e-1
    track_running_stats: bool=True
    affine: bool=True
    device: str="cpu"
    dtype: torch.dtype= torch.float32

    def produce(self, in_size: torch.Size) -> Operation:
        return Operation(self.torch_normalizer_cls(
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
            torch.Size([in_size[0], self.out_features])
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

    dropout_cls: typing.Type[nn.Module] = nn.Dropout
    p: float=0.2
    inplace: bool=False

    def produce(self, in_size: torch.Size) -> Operation:
        return Operation(
            self.dropout_cls(p=self.p, inplace=self.inplace),
            torch.Size(in_size)
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return self.produce(in_size)


@dataclass
class DimAggregateFactory(OpFactory):

    torch_agg_fn: typing.Callable[[int], torch.Tensor]=torch.mean
    dim: int=1
    index: int=None
    keepdim: bool=False

    def produce(self, in_size: torch.Size) -> Operation:
        f = lambda x: (
            self.torch_agg_fn(x, dim=self.dim, keepdim=self.keepdim) if self.index is None
            else self.torch_agg_fn(x, dim=self.dim)[self.index]
        )
        if self.keepdim:
            out_size = in_size[:self.dim] + torch.Size([1]) + in_size[self.dim + 1:]
        else:
            out_size = in_size[:self.dim] + in_size[self.dim + 1:]
        return Operation(
            util_modules.Lambda(f), out_size
        )


@dataclass
class ConvolutionFactory(OpReversibleFactory):
    """For producing convolutional layers. Note: Cannot reverse all configurations."""
    out_features: int=UNDEFINED()
    k: typing.Union[int, typing.Tuple]=1
    stride: typing.Union[int, typing.Tuple]=1
    padding: typing.Union[int, typing.Tuple]=0
    torch_conv_cls: typing.Type[nn.Module]= nn.Conv2d
    torch_deconv_cls: typing.Type[nn.Module]= nn.ConvTranspose2d
    kwargs: dict=field(default_factory=dict)

    @check_undefined
    def produce(self, in_size: torch.Size):

        out_sizes = utils.calc_conv_out_size(in_size, self.k, self.stride, self.padding)
        out_size = torch.Size([-1, self.out_features, *out_sizes])
        return Operation(
            self.torch_conv_cls(in_size[1], self.out_features, 
            self.k, self.stride, padding=self.padding, **self.kwargs),
            out_size
        )
   
    @check_undefined 
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        out_sizes = torch.Size([
            -1, self.out_features, *utils.calc_conv_out_size(in_size, self.k, self.stride, self.padding)
        ])
        in_sizes_comp = torch.Size([
            -1, in_size[1], *utils.calc_conv_transpose_out_size(out_sizes, self.k, self.stride, self.padding)
        ])

        if in_size[1:] != in_sizes_comp[1:]:
            raise ValueError(f"Failed reverse expect: {in_size} actual: {in_sizes_comp} for {out_sizes}")

        return Operation(
            self.torch_deconv_cls(self.out_features, in_size[1],  
            kernel_size=self.k, stride=self.stride, padding=self.padding, **self.kwargs),
            in_size
        )


# TODO: Add DECONV/UNPOOL Factories etc

@dataclass
class PoolFactory(OpReversibleFactory):
    """For producing pooling layers and reverse layers. Note: Cannot reverse all configurations"""

    k: typing.Union[int, typing.Tuple]=1
    stride: typing.Union[int, typing.Tuple]=1
    padding: typing.Union[int, typing.Tuple]=0
    torch_pool_cls: typing.Type[nn.Module]= nn.MaxPool2d
    torch_unpool_cls: typing.Type[nn.Module]= nn.MaxUnpool2d
    kwargs: dict=field(default_factory=dict)

    def produce(self, in_size: torch.Size):
        
        out_sizes = utils.calc_pool_out_size(in_size, self.k, self.stride, self.padding)
        out_size = torch.Size([-1, in_size[1], *out_sizes])
        return Operation(
            self.torch_pool_cls(in_size[1], in_size[1], self.k, self.stride, padding=self.padding),
            out_size
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        out_sizes = utils.calc_maxunpool_out_size(in_size, self.k, self.stride, self.padding)
        out_size = torch.Size([-1, in_size[1], *out_sizes])
        return Operation(
            self.torch_unpool_cls(in_size[1], in_size[1], self.k, self.stride, padding=self.padding),
            out_size
        )
    

@dataclass
class ViewFactory(OpReversibleFactory):
    """For changing the 'view' of the output"""

    view: torch.Size

    def produce(self, in_size: torch.Size):
        
        return Operation(
            util_modules.View(self.view),
            self.view
        )
    
    def produce_reverse(self, in_size: torch.Size) -> Operation:
        return Operation(
            util_modules.View(in_size),
            in_size
        )


@dataclass
class RepeatFactory(OpFactory):
    """For repeating certain dimensions of a network"""

    repeat_by: typing.List[int]
    keepbatch: bool=True

    def produce(self, in_size: torch.Size):
        repeat_by = [*self.repeat_by]
        if self.keepbatch:
            repeat_by.insert(0, 1)

        out_size = []
        for x, y in zip(in_size, repeat_by):
            if x == -1:
                out_size.append(-1)
            else:
                out_size.append(x * y)

        return Operation(
            util_modules.Lambda(lambda x: x.repeat(*repeat_by)), 
            torch.Size(out_size)
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

    torch_loss_cls: typing.Type[nn.Module]= nn.MSELoss
    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    
    def produce(self, in_size: torch.Size):

        return Operation(
            self.torch_loss_cls(reduction=self.reduction_cls.as_str()),
            self.reduction_cls.get_out_size(in_size)
        )


@dataclass
class RegularizerFactory(OpFactory):
    
    regularizer_cls: typing.Type[objectives.Regularizer]= objectives.L2Reg
    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
 
    def produce(self, in_size: torch.Size):

        return Operation(
            self.regularizer_cls(reduction_cls=self.reduction_cls),
            self.reduction_cls.get_out_size(in_size)
        )


@dataclass
class LossFactory(OpFactory):

    loss_cls: typing.Type[objectives.Loss]
    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    
    def produce(self, in_size: torch.Size):

        return Operation(
            self.loss_cls(reduction_cls=self.reduction_cls),
            self.reduction_cls.get_out_size(in_size)
        )


@dataclass
class ValidationFactory(OpFactory):

    validation_cls: typing.Type[objectives.Loss]=objectives.ClassificationFitness
    reduction_cls: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction
    
    def produce(self, in_size: torch.Size):

        return Operation(
            self.validation_cls(reduction_cls=self.reduction_cls),
            self.reduction_cls.get_out_size(in_size)
        )


@dataclass
class AggregateFactory(OpFactory):

    to_average: bool=True
    weights: typing.List=None

    def produce(self, in_size: torch.Size):

        def aggregator(*x):
            if self.weights is not None:
                x = [
                    el * w for el, w in zip(x, self.weights)
                ]
            
            summed = sum(x)
            if self.to_average:
                summed = summed / len(x)
            return summed

        return Operation(
            util_modules.Lambda(aggregator),
            in_size
        )


@dataclass
class LinearLayerDirector(OpDirector):

    in_features: int=UNDEFINED()
    out_features: int=UNDEFINED()
    activation: Opt[ActivationFactory]=ActivationFactory(torch.nn.ReLU)
    dropout: Opt[DropoutFactory]=DropoutFactory()
    normalizer: Opt[NormalizerFactory]=NormalizerFactory()
    bias: bool=True
    device: str="cpu"

    @check_undefined
    def produce(self) -> Operation:
        sequential = nn.Sequential()
        in_size = torch.Size([-1, self.in_features])
        out_size = torch.Size([-1, self.out_features])
        if self.dropout is not None:
            sequential.add_module(self.dropout.produce(in_size).op)
        sequential.add_module(
            nn.Linear(self.in_features, self.out_features, self.bias, self.device)
        )
        
        if self.normalizer is not None:
            sequential.add_module(
                self.normalizer.produce(out_size),
            )
        if self.activation is not None:
            sequential.add_module(
                self.activation.produce(out_size)
            )
        return Operation(sequential, out_size)


@dataclass
class FeedForwardDirector(NetDirector):
    """Use to create a feed forward network"""

    in_features: int=UNDEFINED()
    out_features: typing.List[int]=UNDEFINED()
    base_name: str="layer"
    activation: ActivationFactory=ActivationFactory(torch_act_cls=nn.ReLU)
    out_activation: ActivationFactory=ActivationFactory(torch_act_cls=nn.ReLU)
    normalizer: Opt[NormalizerFactory]=None
    dropout: Opt[DropoutFactory]=None
    input_name: str="x"
    output_name: str="x"
    labels: Opt[typing.List[str]]=None
    use_bias: bool=True
    device: str="cpu"

    @property
    def n_layers(self) -> typing.Union[int, UNDEFINED]:
        if self.out_features == UNDEFINED:
            return UNDEFINED
        return len(self.out_features)
    
    @check_undefined
    def append(self, base_network: BaseNetwork) -> Network:
        """Construct the feedforward network on top of another
        network

        Args:
            base_network (BaseNetwork): Network to build off of

        Returns:
            Network: 
        """
        
        constructor = base_network.constructor
        port, = base_network.ports
        
        linear_factory = LinearFactory(UNDEFINED, self.use_bias, device=self.device)         
        linear_factory.bias = self.use_bias
        linear_factory.device = self.device
        constructor.base_labels = self.labels

        for i, n_out in enumerate(self.out_features):
            linear_factory.out_features = n_out
            if self.dropout is not None: 
                port, = constructor.add_op(
                    f"dropout_{i}", self.dropout.produce(port.size)
                )
            port, = constructor.add_op(
                f"linear_{i}", linear_factory.produce(port.size)
            )
            if self.normalizer is not None:
                port, = constructor.add_op(
                    f"normalizer_{i}", self.normalizer.produce(port.size)
                )
            if i < len(self.out_features  - 1):
                port, = constructor.add_op(
                    f"activation_{i}", self.activation.produce(port.size)
                )
                
        constructor.add_op(     
            self.output_name, self.out_activation.produce(port.size)
        )
        return constructor.net

    def produce(self) -> Network:
        """Produce a new feed forward network

        Returns:
            Network: Network being builtSs
        """

        constructor = NetworkBuilder()
        in_ = constructor.add_tensor_input(
            self.input_name, torch.Size([-1, self.in_features]), 
            labels=self.labels,
            device=self.device
        )
        return self.append(BaseNetwork(constructor, in_))
