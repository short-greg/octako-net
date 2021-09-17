from .builders import LossBuilder
import typing
import torch.nn as nn
import torch
from . import networks
import torch.optim
import torch.nn
from .assemblers import FeedForwardAssembler, FeedForwardLossAssembler
from abc import ABC, abstractmethod

"""Classes related to learning machines

Learning Machines define an operation, a learning algorithm
for learning that operation and a testing algorithm for testing the operation.
"""


def tensor_device(f):
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return {
            k: t.cpu() for k, t in result.items()
        }
    return wrapper


def tensor_device2(f):
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return result.cpu()

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

        # self._interface = networks.NetworkInterface(
        #    network, [agg_loss_name, validation_name, *loss_names], by=[x_name]
        # )
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

        # self._interface = networks.NetworkInterface(
        #    network, [agg_loss_name, validation_name, *loss_names], by=[x_name]
        # )
        self._agg_loss_name = agg_loss_name
        self._validation_name = validation_name
        self._loss_names = loss_names
        self._target_name = target_name
        self._x_name = x_name

    def step(self, x, t):
        
        return self._network.probe(
            [self._agg_loss_name, self._validation_name, *self._loss_names], by={self._x_name: x, self._target_name: t}
        )

        """[summary]

        assembler.set_loss() <- set the weight here and te weight
        assembler.add_output_regularization() <- set the base names and the weight here
        assembler.set_validation() <- set to binary classify.. need to set the name also
        assembler.set_aggregate_loss_name() 
        assembler.loss_names <- retrieve the loss names

        loss_assembler.
        """


class BinaryClassifier2(Learner):

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
        self._loss_builder = loss_builder

        self._loss_builder.set_loss(loss_builder.bce)
        self._loss_builder.set_validator(loss_builder.binary_classifier)
        self._loss_assembler = FeedForwardLossAssembler(
            1, self._loss_builder, self._network_assembler
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
    
    @tensor_device2
    def classify(self, x):
        p = self._classification_interface.forward(x)
        return (p >= 0.5).float().to(self._device)

    @tensor_device
    def learn(self, x, t):
        return self._learning_algorithm.step(x, t)

    @tensor_device
    def test(self, x, t):
        return self._testing_algorithm.step(x, t)

    @property
    def maximize_validation(self):
        return True


class Multiclass2(Learner):

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

        # self._loss_builder.set_loss(loss_builder.cross_entropy)
        # self._loss_builder.set_validator(loss_builder.multiclassifier)
        self._loss_assembler = FeedForwardLossAssembler(
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
    
    @tensor_device2
    def classify(self, x):
        p = torch.nn.Softmax(self._classification_interface.forward(x), dim=1)(x)
        return torch.argmax(p, dim=1)

    @tensor_device
    def learn(self, x, t):
        return self._learning_algorithm.step(x, t)

    @tensor_device
    def test(self, x, t):
        return self._testing_algorithm.step(x, t)

    @property
    def maximize_validation(self):
        return True


class Regressor2(Learner):

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

        # self._loss_builder.set_loss(loss_builder.mse)
        # self._loss_builder.set_validator(loss_builder.mse)
        self._loss_assembler = FeedForwardLossAssembler(
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
    
    @tensor_device2
    def regress(self, x):
        return self._regression_interface.forward(x)

    @tensor_device
    def learn(self, x, t):
        return self._learning_algorithm.step(x, t)

    @tensor_device
    def test(self, x, t):
        return self._testing_algorithm.step(x, t)
    
    @property
    def maximize_validation(self):
        return False



# class RegressorLearner(Learner):

#     def __init__(
#         self, network_assembler: FeedForwardAssembler,
#         loss_factory: nn.Module=nn.MSELoss,
#         optim_factory: torch.optim.Optimizer=torch.optim.Adam, 
#         lr: float=1e-3, device='cpu'
#     ):
#         self._out_name = 'y'
#         self._network_assembler = network_assembler
#         self._network_assembler.set_input_name('x')
#         self._network_assembler.set_output_name('y')
#         self._network = self._network_assembler.build()
#         self._network.to(device)
#         self._optim: torch.optim.Optimizer = optim_factory(self._network.parameters(), lr=lr)
#         self._loss = loss_factory()
#         self._output_interface = networks.NetworkInterface(
#             self._network, self._network.get_ports(self._out_name)
#         )
#         self._device = device

#     @property
#     def fields(self):
#         return ['Loss', 'Classification']

#     @tensor_device2
#     def classify(self, x):
#         y = self._output_interface.forward(x)
#         p = torch.sigmoid(y)
#         return (p >= 0.5).float()

#     @tensor_device
#     def learn(self, x, t):
#         x = x.to(self._device)
#         t = t.to(self._device)
#         self._optim.zero_grad()
#         y, = self._output_interface(x)
#         loss: torch.Tensor = self._loss(y, t)
#         loss.backward()
#         self._optim.step()
#         return {'Loss': loss.to('cpu')}

#     @tensor_device
#     def test(self, x, t):
#         x = x.to(self._device)
#         t = t.to(self._device)
#         y, = self._output_interface(x)
#         loss: torch.Tensor = self._loss(y, t).to('cpu')
#         return {'Loss': loss}


# class MulticlassifierLearner(Learner):

#     def __init__(
#         self, network_assembler: FeedForwardAssembler,
#         loss_factory: nn.Module=nn.CrossEntropyLoss,
#         optim_factory: torch.optim.Optimizer=torch.optim.Adam, 
#         lr: float=1e-3, device='cpu'
#     ):
#         self._out_name = 'y'
        
#         self._network_assembler = network_assembler
#         self._network_assembler.set_input_name('x')
#         self._network_assembler.set_output_name('y')
#         self._network = self._network_assembler.build()
#         self._network.to(device)
#         self._optim: torch.optim.Optimizer = optim_factory(self._network.parameters(), lr=lr)
#         self._loss = loss_factory()
#         self._output_interface = networks.NetworkInterface(
#             self._network, self._network.get_ports(self._out_name)
#         )
#         self._device = device

#     @property
#     def fields(self):
#         return ['Loss', 'Classification']

#     @tensor_device2
#     def regress(self, x):
#         y, = self._output_interface(x)
#         return torch.argmax(torch.nn.Softmax(y, dim=1), dim=1)

#     @tensor_device
#     def learn(self, x, t):
#         self._optim.zero_grad()
#         y, = self._output_interface(x)
#         classification = torch.argmax(torch.nn.Softmax(y, dim=1), dim=1)
#         classification_rate = (classification == t).float().sum() / len(classification)

#         loss: torch.Tensor = self._loss(y, t)

#         loss.backward()
#         self._optim.step()
#         return {'Loss': loss, 'Classification': classification_rate}

#     @tensor_device
#     def test(self, x, t):
#         y, = self._output_interface(x)
#         classification = torch.argmax(torch.nn.Softmax(y, dim=1), dim=1)
#         classification_rate = (classification == t).float().sum() / len(classification)
#         loss: torch.Tensor = self._loss(y, t)
#         return {'Loss': loss, 'Classification': classification_rate}


# class BinaryClassifierLearner(Learner):

#     def __init__(
#         self, network_assembler: FeedForwardAssembler,
#         loss_factory: nn.Module=nn.BCELoss,
#         optim_factory: torch.optim.Optimizer=torch.optim.Adam, 
#         lr: float=1e-3, device='cpu'
#     ):
#         self._out_name = 'y'
#         network_assembler.set_input_name("x").set_output_name(self._out_name)
        
#         self._network_assembler = network_assembler
#         self._network = self._network_assembler.build()
#         self._network.to(device)
#         for p in self._network.parameters():
#             print(p.size())
#         self._optim: torch.optim.Optimizer = optim_factory(self._network.parameters(), lr=lr)
#         self._loss = loss_factory()
#         self._output_interface = networks.NetworkInterface(
#             self._network, self._network.get_ports(self._out_name), by=["x"]
#         )
#         self._device = device

#     @property
#     def fields(self):
#         return ['Loss', 'Classification']
    
#     @tensor_device2
#     def classify(self, x):
#         y = self._output_interface.forward(x)
#         p = torch.sigmoid(y)
#         return (p >= 0.5).float().to(self._device)

#     @tensor_device
#     def learn(self, x, t):
#         self._optim.zero_grad()
#         y = self._output_interface(x).view(-1)
#         p = torch.sigmoid(y)
#         loss: torch.Tensor = self._loss(p, t)
#         loss.backward()
#         self._optim.step()
#         classification = ((p >= 0.5).float() == t).float().mean().to(self._device)

#         return {'Loss': loss, 'Classification': classification}

#     @tensor_device
#     def test(self, x, t):
#         y, = self._output_interface(x)
#         p = torch.sigmoid(y)
#         loss: torch.Tensor = self._loss(p, t)
#         classification = ((p >= 0.5).float() == t).float().mean()

#         return {'Loss': loss, 'Classification': classification}