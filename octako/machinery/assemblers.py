from abc import ABC, abstractmethod
from functools import partial
from itertools import chain

from torch._C import R, contiguous_format

from octako.machinery import utils
from . import builders
from .networks import In, NetworkConstructor, Operation, Port
import typing
import torch
import typing
from .networks import In, Network
from .modules import CompoundLoss, View, Concat
from dataclasses import dataclass


class IAssembler(ABC):
    '''
    Interface for building a network
    '''

    @abstractmethod
    def build(self) -> Network:
        """Build network based on current parameter configuration

        Returns:
            Network: [description]
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, reset_defaults=False):
        """Reset the network being built

        Args:
            reset_defaults (bool, optional): Whether to reset parameters to default values. Defaults to False.
        """
        raise NotImplementedError


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


class FeedForwardAssembler(IAssembler):
    """Assembler for building a feedforward network
    """

    default_input_name = 'x'
    default_output_name = 'y'
    default_network_name = 'network'
    default_dense_name = 'dense'
    default_activation_name = 'activation'

    def __init__(self):
        self._input_name = self.default_input_name
        self._output_name = self.default_output_name
        self._activation_name = self.default_activation_name
        self._dense_name = self.default_dense_name
        self._network_name = self.default_network_name

    def set_input_name(self, name: str):
        """Set input layer name of the network
        """
        self._input_name = name
        return self

    def set_output_name(self, name: str):
        """Set output layer name of the network
        """
        self._output_name = name
        return self

    def reset(self, reset_defaults=False):
        if reset_defaults:
            self._input_name = self.default_input_name
            self._output_name = self.default_output_name
            self._network_name = self.default_network_name
            self._activation_name = self.default_activation_name


class DenseFeedForwardAssembler(FeedForwardAssembler):
    """Assembler for building a standard feedforward network with dense layers
    """

    def __init__(
        self, input_size: int, layer_sizes: typing.List[int], 
        out_size: int
    ):
        """[summary]

        Args:
            feedforward_builder (builders.FeedForwardBuilder): [description]
            input_size (int): [description]
            layer_sizes (typing.List[int]): [description]
            out_size (int): [description]
        """
        super().__init__()
        self.BUILDER = builders.FeedForwardBuilder()
        self._input_size = input_size
        self._layer_sizes = layer_sizes
        self._out_size = out_size
        self._activation = self.BUILDER.relu
        self._normalizer = self.BUILDER.batch_normalizer
        self._dense = self.BUILDER.linear
        self._out_activation = self.BUILDER.sigmoid
    
    def set_activation(self, activation: typing.Callable[[], Operation], **kwargs):
        self._activation = partial(activation, **kwargs)
        return self

    def set_out_activation(self, activation: typing.Callable[[], Operation], **kwargs):
        self._out_activation = partial(activation, **kwargs)
        return self

    def set_normalizer(self, normalizer: typing.Callable[[int], Operation], **kwargs):
        self._normalizer = partial(normalizer, **kwargs)
        return self

    def set_dense(self, dense: typing.Callable[[int, int], Operation], **kwargs):
        self._dense = partial(dense, **kwargs)
        return self

    def reset(
        self, input_size: int=None, layer_sizes: typing.List[int]=None, 
        out_size: int=None, reset_defaults=False
    ):
        """Reset parameters of the network

        Args:
            input_size (int, optional): Update the input size if not None. Defaults to None.
            layer_sizes (typing.List[int], optional): Update the layer sizes if not None. Defaults to None.
            out_size (int, optional): Update the out size if not None. Defaults to None.
            reset_defaults (bool, optional): Reset to default parameters if not None. Defaults to False.
        """
        super().reset(reset_defaults)
        
        self._input_size = utils.coalesce(input_size, self._input_size)
        self._layer_sizes = utils.coalesce(layer_sizes, self._layer_sizes)
        self._out_size = utils.coalesce(out_size, self._out_size)

    def build(self) -> Network:
        """Build the network based on the parameters

        Returns:
            Network:
        """
        constructor = NetworkConstructor(Network())
        in_ = constructor.add_tensor_input(
            self._input_name, torch.Size([-1, self._input_size])
        )
        base_network = self.append(BaseNetwork(constructor.net, in_))
        constructor.net.set_default_interface(
            in_, base_network.ports
        )
        return constructor.net

    def append(self, base_network: BaseNetwork) -> BaseNetwork:

        constructor = base_network.constructor
        cur_in, = base_network.ports
        for i, layer_size in enumerate(self._layer_sizes):

            cur_in, = constructor.add_op(
                f"{self._dense_name}_{i}", self._dense(cur_in.size[1], layer_size),
                cur_in
            )

            cur_in, = constructor.add_op(
                f"{self._activation_name}_{i}", self._activation(cur_in.size), cur_in,
            )
        cur_in, = constructor.add_op(
            f'{self._dense_name}_out', self._dense(cur_in.size[1], self._out_size), cur_in
        )

        cur_in = constructor.add_op(
            self._output_name, self._out_activation(cur_in.size), cur_in
        )
        
        return BaseNetwork(constructor, cur_in)


class ObjectiveAssembler(IAssembler):
    """Assembler for building a feedforward network
    """
    default_label = 'Objective'
    default_objective_name = 'objective'
    default_input_name = 'input'
    DEFAULT_INPUT_SIZE = torch.Size([1,1])

    def __init__(self, input_size: torch.Size=DEFAULT_INPUT_SIZE):

        self.label = self.default_label
        self.objective_name = self.default_objective_name
        self.input_name = self.default_input_name
        self._input_size = input_size

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, input_size: torch.Size):
        self._input_size = input_size

    def set_input_name(self, name: str):
        """Set input layer name of the network
        """
        self._input_name = name
        return self

    def set_output_name(self, name: str):
        """Set output layer name of the network
        """
        self._output_name = name
        return self

    def reset(self,input_size: torch.Size=None,  reset_defaults=False):
        self._input_size = utils.coalesce(input_size, self._input_size)
        if reset_defaults:
            self.label = self.default_label
            self._objective_name = self.default_objective_name
            self._input_name = self.default_input_name


class TargetObjectiveAssembler(ObjectiveAssembler):

    default_target_name = 'target'
    DEFAULT_TARGET_SIZE = torch.Size([1,1])

    def __init__(self, input_size: torch.Size=ObjectiveAssembler.DEFAULT_INPUT_SIZE, target_size: torch.Size=DEFAULT_TARGET_SIZE):

        super().__init__(input_size)
        self.BUILDER = builders.LossBuilder()
        self._target_size = target_size
        self.target_name = self.default_target_name
        self.set_objective(self.BUILDER.mse)
    
    def reset(self, input_size: torch.Size=None, target_size: torch.Size=None, reset_defaults: bool=False):

        super().reset(input_size)
        self._target_size = utils.coalesce(target_size, self._target_size)

        if reset_defaults:
            self.target_name = self.default_target_name

    @property
    def target_size(self):
        return self._target_size

    @target_size.setter
    def target_size(self, target_size: torch.Size):
        self._target_size = target_size

    def set_objective(self, objective: typing.Callable[[int, float], Operation], **kwargs):
        self._objective = partial(objective, **kwargs)
        return self
    
    def build(self) -> Network:
        constructor = NetworkConstructor(Network())
        t, = constructor.add_tensor_input(
            self.target_name, self._target_size,
            labels=[self.label]
        )
        in_, = constructor.add_tensor_input(
            self.input_name, self._input_size,
            labels=[self.label]
        )
        base_network = self.append(BaseNetwork(constructor, [in_, t]))
        constructor.net.set_default_interface([in_, t], base_network.ports)

    def append(self, base_network: BaseNetwork) -> BaseNetwork:

        constructor = base_network.constructor
        in_, t = base_network.ports

        out = constructor.add_op(
            self.objective_name, self._loss(in_.size), [in_, t],
            labels=[self.label]
        )

        return BaseNetwork(constructor, out)


class RegularizerObjectiveAssembler(ObjectiveAssembler):
    
    def __init__(self, input_size: torch.Size=ObjectiveAssembler.DEFAULT_INPUT_SIZE):

        super().__init__(input_size)
        self.BUILDER = builders.LossBuilder()
        self.set_objective(self.BUILDER.l2_reg)

    def set_objective(self, regularizer: typing.Callable[[int, float], Operation], **kwargs):
        self._regularizer = partial(regularizer, **kwargs)
        return self

    def build(self) -> Network:
        constructor = NetworkConstructor(Network())
        in_ = constructor.add_tensor_input(
            self.input_name, self._input_size,
            labels=[self.label]
        )
        network, out_ = self.append(BaseNetwork(constructor, in_))
        network.set_default_interface(in_, out_)
        return network

    def append(self, base_network: BaseNetwork) -> typing.Tuple[Network, Port]:

        constructor = base_network.constructor
        in_, t = base_network.ports
    
        out = constructor.add_op(
            self.objective_name, self._loss(in_.size), [in_, t],
            labels=[self.label]
        )
        return constructor.net, out


class CompoundLossAssembler(IAssembler):

    default_label = 'Loss Sum'
    default_input_name_base = 'input'
    sum_name = 'sum'

    def __init__(self):
        self.input_name = self.default_input_name_base
        self.label: str = self.default_label
        builder = builders.LossBuilder()
        self._aggregator = builder.sum
        self._loss_assemblers: typing.List[typing.Tuple[TargetObjectiveAssembler, float]] = []
        self._regularizer_assemblers: typing.List[typing.Tuple[RegularizerObjectiveAssembler, float]] = []
    
    def add_loss_assembler(self, loss_assembler: TargetObjectiveAssembler, weight: float=1.0):

        self._loss_assemblers.append((loss_assembler, weight))

    def add_regularizer_assembler(self, regularizer_assembler: RegularizerObjectiveAssembler, weight: float=1.0):

        self._regularizer_assemblers.append((regularizer_assembler, weight))
    
    def reset(self, reset_default: bool=False):

        if reset_default:
            self.input_name = self.default_input_name_base
            self.label = self.default_label
        
    @property
    def n_losses(self):
        return len(self._loss_assemblers)
    
    @property
    def n_regularizers(self):
        return len(self._regularizer_assemblers)

    def build(self) -> Network:
        constructor = NetworkConstructor(Network())
        ports = []
        for i in range(len(self._loss_assemblers)):
            in_, = constructor.add_tensor_input(
                self.input_name + '_loss_' + i, torch.Size([]),
                labels=[self.label]
            )
            target, = constructor.add_tensor_input(
                self.input_name + '_' + i, torch.Size([]),
                labels=[self.label]
            )
            ports.append(in_, target)
        for i, _ in range(len(self._regularizer_assemblers)):
            in_, = constructor.add_tensor_input(
                self.input_name + '_regularizer_' + i, torch.Size([]),
                labels=[self.label]
            )
            ports.append(in_)

        base_network = self.append(BaseNetwork(constructor, ports))
        constructor.net.set_default_interface(in_, base_network.ports)
        return constructor.net

    def append(self, base_network: BaseNetwork) -> BaseNetwork:

        constructor = base_network.constructor
        ports = base_network.ports

        weights = []
        i = 0
        for loss_assembler, weight in self._loss_assemblers:
            loss_assembler.append(BaseNetwork(constructor, ports[i:i+2]))
            weights.append(weight)
            i += 2

        for regularizer_assembler, weight in self._regularizer_assemblers:
            regularizer_assembler.append(BaseNetwork(constructor, ports[i:i+1]))
            weights.append(weight)
            i += 1

        out = constructor.add_op(
            self.sum_name, self._aggregator(weights), ports,
            labels=[self.label]
        )

        return BaseNetwork(constructor, out)
