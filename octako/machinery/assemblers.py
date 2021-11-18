from abc import ABC, abstractmethod
from functools import partial
from octako.machinery import utils
from . import builders
from .networks import NetworkConstructor, Operation, Port, Network
import typing
import torch
import typing
from dataclasses import MISSING, asdict, dataclass, field, fields, make_dataclass


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
class BaseNetwork(object):
    """
    Base network to build off of. Can be used in the build() method
    """

    constructor: NetworkConstructor
    ports: typing.List[Port]

    def __post_init__(self):

        if isinstance(self.ports, Port):
            self.ports = [self.ports]


@dataclass
class IAssembler(ABC):
    '''
    Interface for building a network
    '''
    input_name: str="x"
    output_name: str="y"
    input_size: torch.Size=UNDEFINED()

    def __post_init__(self):
        for k, v in asdict(self).items():
            if (v == _UNDEFINED): raise TypeError(f"Missing required argument {k}.")

    def clone(self):
        res = {}
        for k, v in asdict(self).items():
            if isinstance(v, IAssembler):
                res[k] = v.clone()
            else:
                res[k] = v
        return self.__class__(**res)

    def spawn(self):
        res = {}
        for k, v in asdict(self).items():
            if isinstance(v, IAssembler):
                res[k] = v.spawn()
            else:
                cls_v = self.__class__.__getattribute__(k)
                res[k] = v if cls_v == UNDEFINED else cls_v
        return self.__class__(**res)

    @abstractmethod
    def build(self) -> Network:
        """Build network based on current parameter configuration

        Returns:
            Network: [description]
        """
        pass
    
    @abstractmethod
    def append(self) -> BaseNetwork:
        """
        """
        pass


@dataclass
class FeedForwardAssembler(IAssembler):
    """Assembler for building a feedforward network
    """
    network_name = 'network'
    dense_name = 'dense'


FEED_FORWARD_BUILDER = builders.FeedForwardBuilder()

@dataclass
class DenseFeedForwardAssembler(FeedForwardAssembler):
    """Assembler for building a standard feedforward network with dense layers
    """
    
    activation_name: str="act"
    out_size: int=UNDEFINED()
    layer_sizes: typing.List[int]=UNDEFINED()
    dense: typing.Callable[[], Operation] = FEED_FORWARD_BUILDER.linear
    activation: typing.Callable[[], Operation]=FEED_FORWARD_BUILDER.relu
    normalizer: typing.Callable[[float], Operation]=FEED_FORWARD_BUILDER.batch_normalizer
    out_activation: typing.Callable[[], Operation]=FEED_FORWARD_BUILDER.sigmoid

    def build(self) -> Network:
        """Build the network based on the parameters

        Returns:
            Network:
        """
        constructor = NetworkConstructor(Network())
        in_ = constructor.add_tensor_input(
            self.input_name, torch.Size([-1, self.input_size])
        )
        base_network = self.append(BaseNetwork(constructor, in_))
        constructor.net.set_default_interface(
            in_, base_network.ports
        )
        return constructor.net

    def append(self, base_network: BaseNetwork) -> BaseNetwork:

        constructor = base_network.constructor
        cur_in, = base_network.ports
        for i, layer_size in enumerate(self.layer_sizes):

            cur_in, = constructor.add_op(
                f"{self.dense_name}_{i}", self.dense(cur_in.size[1], layer_size),
                cur_in
            )

            cur_in, = constructor.add_op(
                f"{self.activation_name}_{i}", self.activation(cur_in.size), cur_in,
            )
        cur_in, = constructor.add_op(
            f'{self.dense_name}_out', self.dense(cur_in.size[1], self.out_size), cur_in
        )

        cur_in = constructor.add_op(
            self.output_name, self.out_activation(cur_in.size), cur_in
        )
        
        return BaseNetwork(constructor, cur_in)
