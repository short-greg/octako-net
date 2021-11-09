from .builders import LossBuilder
import typing
import torch
from . import networks
import torch.optim
import torch.nn
from .assemblers import FeedForwardAssembler, CompoundFeedForwardLossAssembler
from abc import ABC, abstractmethod

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


class BinaryClassifier(Learner):

    def __init__(
        self, network_assembler: FeedForwardAssembler, 
        learning_algorithm_cls: typing.Type[LearningAlgorithm],
        testing_algorithm_cls: typing.Type[TestingAlgorithm],
        optim_factory: typing.Callable[[torch.nn.ParameterList], torch.optim.Optimizer]=torch.optim.Adam, 
        device='cpu'
    ):
        self._out_name = 'y'
        self._target_name = 't'
        self._input_name = 'x'
        self._loss_name = 'loss'
        self._validation_name = 'Classification'

        self._network_assembler = network_assembler
        loss_builder: LossBuilder = LossBuilder()
        # self._loss_assembeler: CompoundFeedForwardLossAssembler = CompoundFeedForwardLossAssembler()

        self._loss_assembler = CompoundFeedForwardLossAssembler(
            1, loss_builder, self._network_assembler
        )
        self._loss_assembler.set_loss(loss_builder.bce)
        self._loss_assembler.set_validator(loss_builder.binary_classifier)

        self._loss_assembler.set_input_name("x").set_output_name(self._out_name).set_loss_name(
            self._loss_name
        ).set_validation_name(
            self._validation_name
        ).set_target_name(self._target_name)
    
        self._network = self._loss_assembler.build()
        self._network.to(device)
        self._learning_algorithm = learning_algorithm_cls(
            optim_factory, self._network, self._input_name, self._target_name, 
            self._loss_name, self._validation_name, self._loss_assembler.loss_names
        )
        self._testing_algorithm = testing_algorithm_cls(
            self._network, self._input_name, self._target_name, self._loss_name, self._validation_name, self._loss_assembler.loss_names
        )
        
        self._classification_interface = networks.NetworkInterface(
            self._network, self._network.get_ports(self._validation_name), by=[self._input_name, self._target_name]
        )
        self._device = device

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

    def __init__(
        self, network_assembler: FeedForwardAssembler, n_classes: int,
        learning_algorithm_cls: typing.Type[LearningAlgorithm],
        testing_algorithm_cls: typing.Type[TestingAlgorithm],
        optim_factory: typing.Callable[[torch.nn.ParameterList], torch.optim.Optimizer]=torch.optim.Adam, 
        device='cpu'
    ):
        self._out_name = 'y'
        self._target_name = 't'
        self._input_name = 'x'
        self._loss_name = 'loss'
        self._validation_name = 'Classification'
        self._n_classes = n_classes

        self._network_assembler = network_assembler
        loss_builder: LossBuilder = LossBuilder()
        self._loss_builder = loss_builder

        self._loss_assembler = CompoundFeedForwardLossAssembler(
            1, self._loss_builder, self._network_assembler
        )
        self._loss_assembler.set_loss(loss_builder.cross_entropy)
        self._loss_assembler.set_validator(loss_builder.multiclassifier)

        self._loss_assembler.set_input_name("x").set_output_name(self._out_name).set_loss_name(
            self._loss_name
        ).set_validation_name(
            self._validation_name
        ).set_target_name(self._target_name)
    
        self._network = self._loss_assembler.build()
        self._network.to(device)

        self._learning_algorithm = learning_algorithm_cls(
            optim_factory, self._network, self._input_name, self._target_name, 
            self._loss_name, self._validation_name, self._loss_assembler.loss_names
        )
        self._testing_algorithm = testing_algorithm_cls(
            self._network, self._input_name, self._target_name, self._loss_name, self._validation_name, self._loss_assembler.loss_names
        )
        
        self._classification_interface = networks.NetworkInterface(
            self._network, self._network.get_ports(self._validation_name), by=[self._input_name, self._target_name]
        )
        self._device = device

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

    def __init__(
        self, network_assembler: FeedForwardAssembler, n_out: int,
        learning_algorithm_cls: typing.Type[LearningAlgorithm],
        testing_algorithm_cls: typing.Type[TestingAlgorithm],
        optim_factory: typing.Callable[[torch.nn.ParameterList], torch.optim.Optimizer]=torch.optim.Adam, 
        device='cpu'
    ):
        self._out_name = 'y'
        self._target_name = 't'
        self._input_name = 'x'
        self._loss_name = 'loss'
        self._validation_name = 'validation'

        self._network_assembler = network_assembler
        loss_builder: LossBuilder = LossBuilder()
        self._loss_builder = loss_builder
        self._n_out = n_out

        self._loss_assembler = CompoundFeedForwardLossAssembler(
            n_out, self._loss_builder, self._network_assembler
        )

        self._loss_assembler.set_loss(loss_builder.mse)
        self._loss_assembler.set_validator(loss_builder.mse)
        self._loss_assembler.set_input_name("x").set_output_name(self._out_name).set_loss_name(
            self._loss_name
        ).set_validation_name(
            self._validation_name
        ).set_target_name(self._target_name)
    
        self._network = self._network_assembler.build()
        self._network.to(device)
        self._learning_algorithm = learning_algorithm_cls(
            optim_factory, self._network, self._input_name, self._target_name, 
            self._loss_name, self._validation_name, self._loss_assembler.loss_names
        )
        self._testing_algorithm = testing_algorithm_cls(
            self._network, self._input_name, self._target_name, self._loss_name, self._validation_name, self._loss_assembler.loss_names
        )
        
        self._regression_interface = networks.NetworkInterface(
            self._network, self._network.get_ports(self._validation_name), by=[self._input_name, self._target_name]
        )
        self._device = device

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
