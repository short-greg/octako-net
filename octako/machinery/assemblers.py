from abc import ABC, abstractmethod

from octako.machinery import utils
from . import builders
from .networks import In, Operation, Port
import typing
import torch
import typing
from .networks import In, Network
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

    network: Network
    ports: typing.List[Port]


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
        builder = builders.FeedForwardBuilder()
        self._input_size = input_size
        self._layer_sizes = layer_sizes
        self._out_size = out_size
        self._activation = builder.relu
        self._normalizer = builder.batch_normalizer
        self._dense = builder.linear
        self._out_activation = builder.sigmoid
    
    def set_activation(self, activation: typing.Callable[[], Operation] ):
        self._activation = activation
        return self

    def set_out_activation(self, activation: typing.Callable[[], Operation] ):
        self._out_activation = activation
        return self

    def set_normalizer(self, normalizer: typing.Callable[[int], Operation]):
        self._normalizer = normalizer
        return self

    def set_dense(self, dense: typing.Callable[[int, int], Operation]):
        self._dense = dense
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

        network = Network()
        network_in, = network.add_node(
            In.from_tensor(self._input_name, torch.Size([-1, self._input_size]))
        )
        cur_in = network_in
        for i, layer_size in enumerate(self._layer_sizes):

            cur_in, = network.add_node(
                f"{self._dense_name}_{i}", self._dense(cur_in.size[1], layer_size),
                cur_in
            )

            cur_in, = network.add_node(
                f"{self._activation_name}_{i}", self._activation(cur_in.size), cur_in,
            )
        cur_in, = network.add_node(
            f'{self._dense_name}_out', self._dense(cur_in.size[1], self._out_size), cur_in
        )

        cur_in, = network.add_node(
            self._output_name, self._out_activation(cur_in.size), cur_in
        )

        network.set_default_interface(
            [network_in], [cur_in]
        )
        # network.set_outputs([cur_in])
        
        return network


# TODO:
# 1) see if any updates needed for loss builder
# 2) add in add_input?
# network([In()]) <- can "add inputs here"
# set_output_interface() <- set the interface in here
# add_input


class FeedForwardLossAssembler(IAssembler):

    default_loss_name = 'loss'
    default_validation_name = 'validation'
    default_regularization_name = 'regularization'

    def __init__(self, target_size: int, loss_builder: builders.LossBuilder, feed_forward_assembler: FeedForwardAssembler):
        super().__init__()
        self._validation_name = self.default_validation_name
        self._loss_name = self.default_loss_name
        self._regularization_name = self.default_regularization_name
        self._feedforward_assembler = feed_forward_assembler
        self._target_name = 't'
        self._target_size = target_size
        self._loss_builder = loss_builder
        self._output_name = 'y'
        self._feedforward_assembler.set_output_name(self._output_name)
        self._input_name = 'x'
        self._feedforward_assembler.set_output_name(self._input_name)
        self._scale_target = False

        base_builder = builders.LossBuilder()
        self._validator = base_builder.mse
        self._target_processor = base_builder.null_processor
        self._regularizer = base_builder.l2_reg
        self._loss = base_builder.mse

    def set_loss_name(self, name: str):
        self._loss_name = name
        return self

    def set_validation_name(self, name: str):
        self._validation_name = name
        return self

    def set_regularization_name(self, name: str):
        self._regularization_name = name
        return self

    def set_input_name(self, name: str):
        """Set input layer name of the network
        """
        self._input_name = name
        self._feedforward_assembler.set_output_name(self._input_name)
        return self

    def set_output_name(self, name: str):
        """Set output layer name of the network
        """
        self._output_name = name
        self._feedforward_assembler.set_output_name(name)
        return self

    def set_target_name(self, name: str):
        """Set name of the target input
        """
        self._target_name = name
        return self

    # def loss(self, in_size: torch.Size, weight: typing.Union[float, torch.Tensor]=1.0):
    #     return self._loss(in_size, weight)

    def set_loss(self, loss: typing.Callable[[int, float], Operation]):
        self._loss = loss
        return self

    def set_regularizer(self, regularizer: typing.Callable[[int, float], Operation]):
        self._regularizer = regularizer
        return self
    
    def set_validator(self, validator: typing.Callable[[int], Operation]):
        self._validator = validator
        return self

    def set_target_processor(self, target_processor:  typing.Callable[[torch.Size], Operation]):
        self._target_processor = target_processor
        return self

    @property
    def loss_names(self):
        return [self._loss_name]

    def reset(
        self, feedforward_assembler: FeedForwardAssembler=None, reset_defaults=False
    ):
        """Reset parameters of the network

        Args:
            input_size (int, optional): Update the input size if not None. Defaults to None.
            layer_sizes (typing.List[int], optional): Update the layer sizes if not None. Defaults to None.
            out_size (int, optional): Update the out size if not None. Defaults to None.
            reset_defaults (bool, optional): Reset to default parameters if not None. Defaults to False.
        """
        super().reset(reset_defaults)

        if feedforward_assembler is not None:
            self._feedforward_assembler = feedforward_assembler
        else:
            self._feedforward_assembler.reset(reset_defaults)

        if reset_defaults:
            self._validation_name = self.default_validation_name
            self._loss_name = self.default_loss_name
            self._regularization_name = self.default_regularization_name

    def build(self) -> Network:
        """Build the network based on the parameters

        Returns:
            Network:
        """

        network = self._feedforward_assembler.build()

        t, = network.add_input(
            In.from_tensor(self._target_name, torch.Size([-1, self._target_size]))
        )
        t, = network.add_node('process target', self._target_processor(t.size), t)
        y, = network.get_ports(self._output_name)

        loss, = network.add_node(
            self._loss_name, self._loss(y.size), [y, t]
        )

        validation, = network.add_node(
            self._validation_name, self._validator(y.size), [y, t]
        )

        network.set_default_interface(
            network.get_ports(self._input_name),
            [loss, validation]
        )

        return network
