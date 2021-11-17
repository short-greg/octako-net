from abc import ABC, abstractmethod
from functools import partial
from octako.machinery import utils
from . import builders
from .networks import NetworkConstructor, Operation, Port, Network
import typing
import torch
import typing
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
    
    @abstractmethod
    def set_names(self):
        pass


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
        self.input_name: str = self.default_input_name
        self.output_name: str = self.default_output_name
        self.activation_name: str = self.default_activation_name
        self.dense_name: str = self.default_dense_name
        self.network_name: str = self.default_network_name

    def set_names(self, input_: str=None, output: str=None):
        self.input_name = utils.coalesce(input_, self.input_name)
        self.output_name = utils.coalesce(output, self.output_name)
        return self

    def set_input_name(self, name: str):
        """Set input layer name of the network
        """
        self.input_name = name
        return self

    def set_output_name(self, name: str):
        """Set output layer name of the network
        """
        self.output_name = name
        return self

    def reset(self, reset_defaults=False):
        if reset_defaults:
            self.input_name = self.default_input_name
            self.output_name = self.default_output_name
            self.network_name = self.default_network_name
            self.activation_name = self.default_activation_name


class DenseFeedForwardAssembler(FeedForwardAssembler):
    """Assembler for building a standard feedforward network with dense layers
    """
    BUILDER = builders.FeedForwardBuilder()

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
        self._input_size = input_size
        self._layer_sizes = layer_sizes
        self._out_size = out_size
        self._activation = self.BUILDER.relu
        self._normalizer = self.BUILDER.batch_normalizer
        self._dense = self.BUILDER.linear
        self._out_activation = self.BUILDER.sigmoid
        self.activation_name = "act"

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

    def set_names(self, input_: str=None, output: str=None, activation: str=None):
        super().set_names(input_, output)
        self.activation_name = utils.coalesce(activation, self.activation_name)
        self.output_name = utils.coalesce(output, self.output_name)
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
            self.input_name, torch.Size([-1, self._input_size])
        )
        base_network = self.append(BaseNetwork(constructor, in_))
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


# class AbstractObjectiveAssembler(IAssembler):
#     """Assembler for building a feedforward network
#     """
#     default_label = 'Objective'
#     default_objective_name = 'objective'
#     default_input_name = 'input'
#     DEFAULT_INPUT_SIZE = torch.Size([1,1])

#     def __init__(self, input_size: torch.Size=DEFAULT_INPUT_SIZE):

#         self.label = self.default_label
#         self.objective_name = self.default_objective_name
#         self.input_name = self.default_input_name
#         self._input_size = input_size

#     def set_names(self, input_: str=None, objective: str=None, label: str=None):
#         self.objective_name = utils.coalesce(objective, self.objective_name)
#         self.label = utils.coalesce(label, self.label)
#         self.input_name = utils.coalesce(input_, self.input_name)
#         return self

#     @property
#     def input_size(self):
#         return self._input_size

#     @input_size.setter
#     def input_size(self, input_size: torch.Size):
#         self._input_size = input_size

#     def prepend_names(self, prepend_with: str):

#         self.objective_name = f'{prepend_with}_{self.objective_name}'
#         self.input_name = f'{prepend_with}_{self.input_name}'

#     def reset(self,input_size: torch.Size=None, reset_defaults=False):
#         self._input_size = utils.coalesce(input_size, self._input_size)
#         if reset_defaults:
#             self.label = self.default_label
#             self.objective_name = self.default_objective_name
#             self.input_name = self.default_input_name


# class TargetAssembler(AbstractObjectiveAssembler):

#     default_target_name = 'target'
#     DEFAULT_TARGET_SIZE = torch.Size([1,1])
#     BUILDER = builders.ObjectiveBuilder()

#     def __init__(self, input_size: torch.Size=AbstractObjectiveAssembler.DEFAULT_INPUT_SIZE, target_size: torch.Size=DEFAULT_TARGET_SIZE):

#         super().__init__(input_size)
#         self._target_size = target_size
#         self.target_name = self.default_target_name
#         self.set_objective(self.BUILDER.mse)

#     def set_names(self, input_: str=None, target: str=None, objective: str=None, label: str=None):
#         super().set_names(input_, objective, label)
#         self.target_name = utils.coalesce(target, self.target_name)
#         return self

#     def reset(self, input_size: torch.Size=None, target_size: torch.Size=None, reset_defaults: bool=False):

#         super().reset(input_size)
#         self._target_size = utils.coalesce(target_size, self._target_size)

#         if reset_defaults:
#             self.target_name = self.default_target_name

#     def prepend_names(self, prepend_with: str):
#         super().prepend_names(prepend_with)
#         self.target_name = f'{prepend_with}_{self.target_name}'

#     @property
#     def target_size(self):
#         return self._target_size

#     @target_size.setter
#     def target_size(self, target_size: torch.Size):
#         self._target_size = target_size

#     def set_objective(self, objective: typing.Callable[[torch.Size, float], Operation], **kwargs):
#         self._objective = partial(objective, **kwargs)
#         return self
    
#     def build(self) -> Network:
#         constructor = NetworkConstructor(Network())
#         t, = constructor.add_tensor_input(
#             self.target_name, self._target_size,
#             labels=[self.label]
#         )
#         in_, = constructor.add_tensor_input(
#             self.input_name, self._input_size,
#             labels=[self.label]
#         )
#         base_network = self.append(BaseNetwork(constructor, [in_, t]))
#         constructor.net.set_default_interface([in_, t], base_network.ports)
#         return constructor.net

#     def append(self, base_network: BaseNetwork) -> BaseNetwork:

#         constructor = base_network.constructor
#         in_, t = base_network.ports

#         out = constructor.add_op(
#             self.objective_name, self._objective(in_.size), [in_, t],
#             labels=[self.label]
#         )

#         return BaseNetwork(constructor, out)


# class RegularizerAssembler(AbstractObjectiveAssembler):
    
#     BUILDER = builders.ObjectiveBuilder()

#     def __init__(self, input_size: torch.Size=AbstractObjectiveAssembler.DEFAULT_INPUT_SIZE):

#         super().__init__(input_size)
#         self.set_objective(self.BUILDER.l2_reg)

#     def set_objective(self, regularizer: typing.Callable[[torch.Size, float], Operation], **kwargs):
#         self._regularizer = partial(regularizer, **kwargs)
#         return self

#     def build(self) -> Network:
#         constructor = NetworkConstructor(Network())
#         in_ = constructor.add_tensor_input(
#             self.input_name, self._input_size,
#             labels=[self.label]
#         )
#         base_network = self.append(BaseNetwork(constructor, in_))
#         constructor.net.set_default_interface(in_, base_network.ports)
#         return constructor.net

#     def append(self, base_network: BaseNetwork) -> BaseNetwork:

#         constructor = base_network.constructor
#         in_, = base_network.ports
    
#         out = constructor.add_op(
#             self.objective_name, self._regularizer(in_.size), [in_],
#             labels=[self.label]
#         )
#         return BaseNetwork(constructor.net, out)


# @dataclass
# class WeightedObjective(object):
#     objective: AbstractObjectiveAssembler
#     name: str
#     weight: float=1.0


# class CompoundLossAssembler(IAssembler):

#     default_label = 'Loss Sum'
#     default_input_name_base = 'input'
#     sum_name = 'sum'

#     def __init__(self):
#         self.input_name = self.default_input_name_base
#         self.label: str = self.default_label
#         builder = builders.ObjectiveBuilder()
#         self._aggregator = builder.sum
#         self._loss_assemblers: typing.List[typing.Tuple[TargetAssembler, float]] = []
#         self._regularizer_assemblers: typing.List[typing.Tuple[RegularizerAssembler, float]] = []
    
#     def set_names(self, input_base: str=None, label: str=None):
#         self.label = utils.coalesce(label, self.label)
#         self.input_name_base = utils.coalesce(input_base, self.input_name_base)
#         return self

#     def add_loss_assembler(self, key: str, loss_assembler: TargetAssembler, weight: float=1.0):

#         loss_assembler.prepend_names(key)
#         self._loss_assemblers.append((loss_assembler, weight))

#     def add_regularizer_assembler(self, key: str, regularizer_assembler: RegularizerAssembler, weight: float=1.0):

#         regularizer_assembler.prepend_names(key)
#         self._regularizer_assemblers.append((regularizer_assembler, weight))
    
#     def reset(self, reset_default: bool=False):

#         if reset_default:
#             self.input_name = self.default_input_name_base
#             self.label = self.default_label
        
#     @property
#     def n_losses(self):
#         return len(self._loss_assemblers)
    
#     @property
#     def n_regularizers(self):
#         return len(self._regularizer_assemblers)

#     def build(self) -> Network:
#         constructor = NetworkConstructor(Network())
#         ports = []

#         for i, (assembler, _) in enumerate(self._loss_assemblers):
#             in_, = constructor.add_tensor_input(
#                 f'{self.input_name}_loss_inp_{i}', assembler.input_size,
#                 labels=[self.label]
#             )
#             target, = constructor.add_tensor_input(
#                 f'{self.input_name}_loss_tar_{i}', assembler.target_size,
#                 labels=[self.label]
#             )
#             ports.extend([in_, target])
#         for i, (assembler, _) in enumerate(self._regularizer_assemblers):
#             in_, = constructor.add_tensor_input(
#                 f'{self.input_name}_reg_inp_{i}', assembler.input_size,
#                 labels=[self.label]
#             )
#             ports.append(in_)

#         base_network = self.append(BaseNetwork(constructor, ports))
#         constructor.net.set_default_interface(ports, base_network.ports)
#         return constructor.net

#     def append(self, base_network: BaseNetwork) -> BaseNetwork:

#         constructor = base_network.constructor
#         ports = base_network.ports

#         weights = []
#         i = 0
#         objective_ports = []
#         for loss_assembler, weight in self._loss_assemblers:
#             base = loss_assembler.append(BaseNetwork(constructor, ports[i:i+2]))
#             objective_ports.extend(base.ports)
#             weights.append(weight)
#             i += 2

#         for regularizer_assembler, weight in self._regularizer_assemblers:
#             base = regularizer_assembler.append(BaseNetwork(constructor, ports[i:i+1]))
#             objective_ports.extend(base.ports)
#             weights.append(weight)
#             i += 1
        
#         out = constructor.add_op(
#             self.sum_name, self._aggregator(weights), objective_ports,
#             labels=[self.label]
#         )

#         return BaseNetwork(constructor, out)
