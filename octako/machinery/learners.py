from .builders import FeedForwardBuilder, ObjectiveBuilder
import typing
import torch
from . import networks
import torch.optim
import torch.nn
from .assemblers import (
    BaseNetwork, DenseFeedForwardAssembler, FeedForwardAssembler, IAssembler
)
from abc import ABC, abstractmethod

from octako.machinery import builders

"""Classes related to learning machines

Learning Machines define an operation, a learning algorithm
for learning that operation and a testing algorithm for testing the operation.
"""

def args_to_device(f):
    # function decorator for converting the args to the 
    # device of the learner
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return result

    return wrapper


def result_to_cpu(f):
    # function decorator for converting the result to cpu
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return result.cpu()

    return wrapper


def dict_result_to_cpu(f):
    # function decorator for converting the results to cpu
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return {
            k: t.cpu() for k, t in result.items()
        }
    return wrapper


class Learner(ABC):
    """Base learning machine class
    """

    @abstractmethod
    def learn(self, x, t):
        """Function for learning the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, x, t):
        """Function for evaluating the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError
    
    @property
    def fields(self):
        raise NotImplementedError


class LearningAlgorithm(ABC):

    @abstractmethod
    def step(self, x, y):
        pass


class TestingAlgorithm(ABC):

    @abstractmethod
    def step(self, x, y):
        pass


class MinibatchLearningAlgorithm(LearningAlgorithm):

    def __init__(self, optim_factory, network: networks.Network, x_name: str, target_name: str, agg_loss_name: str, validation_name: str, loss_names: str):

        self._network = network
        self._optim: torch.optim.Optimizer = optim_factory(self._network.parameters())
        self._agg_loss_name = agg_loss_name
        self._validation_name = validation_name
        self._loss_names = loss_names
        self._target_name = target_name
        self._x_name = x_name

    def step(self, x, t):
    
        self._optim.zero_grad()
        results = self._network.probe(
            [self._agg_loss_name, self._validation_name, *self._loss_names], by={self._x_name: x, self._target_name: t}
        )
        loss: torch.Tensor = results[self._agg_loss_name]
        loss.backward()
        self._optim.step()
        return results


class MinibatchTestingAlgorithm(TestingAlgorithm):

    def __init__(self, network: networks.Network, x_name: str, target_name: str, agg_loss_name: str, validation_name: str, loss_names: str):

        self._network = network
        self._agg_loss_name = agg_loss_name
        self._validation_name = validation_name
        self._loss_names = loss_names
        self._target_name = target_name
        self._x_name = x_name

    def step(self, x, t):
        
        return self._network.probe(
            [self._agg_loss_name, self._validation_name, *self._loss_names], by={self._x_name: x, self._target_name: t}
        )

#TODO: create this assembler
class TeachingNetworkAssembler(IAssembler):

    OBJECTIVE_BUILDER = builders.ObjectiveBuilder

    def __init__(self, network_assembler, taget_size: torch.Size):
        pass

    def set_names(self, input_: str=None, target: str=None, loss: str=None, validation: str=None):
        pass

    def set_loss(self, objective):
        pass

    def set_validation(self, objective):
        pass

    def build(self):
        pass

    def append(self):
        pass


class BinaryClassifier(Learner):

    OUT_NAME = 'y'
    TARGET_NAME = 't'
    INPUT_NAME = 'x'
    LOSS_NAME = 'loss'
    VALIDATION_NAME = 'classification'

    def __init__(
        self, network_assembler: FeedForwardAssembler, 
        learning_algorithm_cls: typing.Type[LearningAlgorithm],
        testing_algorithm_cls: typing.Type[TestingAlgorithm],
        optim_factory: typing.Callable[[torch.nn.ParameterList], torch.optim.Optimizer]=torch.optim.Adam, 
        device='cpu'
    ):
        self._device = device
        self._network = self._assemble_network(network_assembler)

        self._learning_algorithm = learning_algorithm_cls(
            optim_factory, self._network, self.INPUT_NAME, self.TARGET_NAME, 
            self.LOSS_NAME, self.VALIDATION_NAME, [self.LOSS_NAME]
        )
        self._testing_algorithm = testing_algorithm_cls(
            self._network, self.INPUT_NAME, self.TARGET_NAME, self.LOSS_NAME, self.VALIDATION_NAME, [self.LOSS_NAME]
        )
        self._classification_interface = networks.NetworkInterface(
            self._network, [self.INPUT_NAME, self.TARGET_NAME], [self.OUT_NAME]
        )

    def _assemble_network(self, network_assembler: DenseFeedForwardAssembler):

        loss_builder = builders.ObjectiveBuilder()
        target_size = torch.Size([-1])
        network_assembler = network_assembler
        network_assembler.input_name = self.INPUT_NAME
        network_assembler.output_name = self.OUT_NAME
        
        constructor = networks.NetworkConstructor(network_assembler.build())
        in_, out = constructor[[self.INPUT_NAME, self.OUT_NAME]].ports
        target, = constructor.add_tensor_input(self.TARGET_NAME, target_size, labels=["target"])
        
        # out, = constructor.add_op("Flatten Out", feed_forward_builder.flatten(), out, labels=["classifier"])
        
        constructor.add_op(
            self.LOSS_NAME, loss_builder.bce(target_size), [out, target], "loss"
        )
        constructor.add_op(
            self.VALIDATION_NAME, loss_builder.binary_val(target_size), [out, target], "validation"
        )
        
        constructor.net.to(self._device)
        return constructor.net

    @property
    def fields(self):
        return ['Loss', 'Classification']
    
    @args_to_device
    @result_to_cpu
    def classify(self, x):
        p = self._classification_interface.forward(x)
        return (p >= 0.5).float().to(self._device)

    @args_to_device
    @dict_result_to_cpu
    def learn(self, x, t):
        return self._learning_algorithm.step(x, t)

    @args_to_device
    @dict_result_to_cpu
    def test(self, x, t):
        return self._testing_algorithm.step(x, t)

    @property
    def maximize_validation(self):
        return True


class Multiclass(Learner):

    OUT_NAME = 'y'
    TARGET_NAME = 't'
    INPUT_NAME = 'x'
    LOSS_NAME = 'loss'
    VALIDATION_NAME = 'classification'
    FUZZY_OUT_NAME = 'fuzzy_y'

    def __init__(
        self, network_assembler: FeedForwardAssembler, n_classes: int,
        learning_algorithm_cls: typing.Type[LearningAlgorithm],
        testing_algorithm_cls: typing.Type[TestingAlgorithm],
        optim_factory: typing.Callable[[torch.nn.ParameterList], torch.optim.Optimizer]=torch.optim.Adam, 
        device='cpu'
    ):
        self._device = device
        self._network = self._assemble_network(network_assembler)

        self._learning_algorithm = learning_algorithm_cls(
            optim_factory, self._network, self.INPUT_NAME, self.TARGET_NAME, 
            self.LOSS_NAME, self.VALIDATION_NAME, [self.LOSS_NAME]
        )
        self._testing_algorithm = testing_algorithm_cls(
            self._network, self.INPUT_NAME, self.TARGET_NAME, self.LOSS_NAME, self.VALIDATION_NAME, [self.LOSS_NAME]
        )
        
        self._classification_interface = networks.NetworkInterface(
            self._network, [self.INPUT_NAME, self.TARGET_NAME], [self.OUT_NAME]
        )
        self._device = device

    def _assemble_network(self, network_assembler: DenseFeedForwardAssembler):

        builder = builders.ObjectiveBuilder()
        network_assembler.input_name = self.INPUT_NAME
        network_assembler.output_name = self.OUT_NAME

        constructor = networks.NetworkConstructor(network_assembler.build())
        in_, out = constructor[[self.INPUT_NAME, self.FUZZY_OUT_NAME]].ports
        target, = constructor.add_tensor_input(self.TARGET_NAME, out.size, labels=["target"])

        out, = constructor.add_lambda_op(
            self.OUT_NAME, lambda x: torch.log(x), out.size, [out], "out"
        )
        constructor.add_op(
            self.LOSS_NAME, builder.cross_entropy(out.size), [out, target], "loss"
        )
        constructor.add_op(
            self.VALIDATION_NAME, builder.multiclass_val(out.size), [out, target], "validation"
        )
        
        network = constructor.net
        network.to(self._device)
        return network

    @property
    def fields(self):
        return ['Loss', 'Classification']
    
    @args_to_device
    @result_to_cpu
    def classify(self, x):
        p = torch.nn.Softmax(self._classification_interface.forward(x), dim=1)(x)
        return torch.argmax(p, dim=1)

    @args_to_device
    @dict_result_to_cpu
    def learn(self, x, t):
        return self._learning_algorithm.step(x, t)

    @args_to_device
    @dict_result_to_cpu
    def test(self, x, t):
        return self._testing_algorithm.step(x, t)

    @property
    def maximize_validation(self):
        return True


class Regressor(Learner):

    OUT_NAME = 'y'
    TARGET_NAME = 't'
    INPUT_NAME = 'x'
    LOSS_NAME = 'loss'
    VALIDATION_NAME = 'validation'

    def __init__(
        self, network_assembler: DenseFeedForwardAssembler, 
        learning_algorithm_cls: typing.Type[LearningAlgorithm],
        testing_algorithm_cls: typing.Type[TestingAlgorithm],
        optim_factory: typing.Callable[[torch.nn.ParameterList], torch.optim.Optimizer]=torch.optim.Adam, 
        device='cpu'
    ):
        # out_size = torch.Size([-1, n_out])
        self._device = device
        network_assembler.input_name = self.INPUT_NAME
        network_assembler.output_name = self.OUT_NAME
        self._network = self._assemble_network(network_assembler)

        self._learning_algorithm = learning_algorithm_cls(
            optim_factory, self._network, self.INPUT_NAME, self.TARGET_NAME, 
            self.LOSS_NAME, self.VALIDATION_NAME, [self.LOSS_NAME]
        )
        self._testing_algorithm = testing_algorithm_cls(
            self._network, self.INPUT_NAME, self.TARGET_NAME, self.LOSS_NAME, self.VALIDATION_NAME, [self.LOSS_NAME]
        )
        self._classification_interface = networks.NetworkInterface(
            self._network, [self.INPUT_NAME, self.TARGET_NAME], [self.OUT_NAME]
        )

    def _assemble_network(self, network_assembler: DenseFeedForwardAssembler):

        # Can probably put this in the base class

        builder = builders.ObjectiveBuilder()
        network_assembler = network_assembler.set_names(
            input_=self.INPUT_NAME, output=self.OUT_NAME
        )
        constructor = networks.NetworkConstructor(network_assembler.build())
        in_, out = constructor[[self.INPUT_NAME, self.OUT_NAME]].ports
        target = constructor.add_tensor_input(self.TARGET_NAME, out.size, labels=["target"])

        constructor.add_op(
            self.LOSS_NAME, builder.mse(out.size), [out, target], "loss"
        )
        constructor.add_op(
            self.VALIDATION_NAME, builder.mse(out.size), [out, target], "validation"
        )

        constructor.net.to(self._device)
        return constructor.net

    @property
    def fields(self):
        return ['Loss', 'Regression']
    
    @args_to_device
    @result_to_cpu
    def regress(self, x):
        return self._regression_interface.forward(x)

    @args_to_device
    @dict_result_to_cpu
    def learn(self, x, t):
        return self._learning_algorithm.step(x, t)

    @args_to_device
    @dict_result_to_cpu
    def test(self, x, t):
        return self._testing_algorithm.step(x, t)
    
    @property
    def maximize_validation(self):
        return False
