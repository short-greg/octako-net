from .builders import ObjectiveBuilder
import typing
import torch
from . import networks
import torch.optim
import torch.nn
from .assemblers import BaseNetwork, DenseFeedForwardAssembler, FeedForwardAssembler, IAssembler, TargetAssembler, TargetObjectiveAssembler, CompoundLossAssembler, TrainingNetworkAssembler, WeightedObjective
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


# TODO: UPpdate the learners witht he new losses

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
            self._network, self.INPUT_NAME, self.TARGET_NAME, self.LOSS_NAME, self.VALIDATION_NAME, self._loss_assembler.loss_names
        )
        self._classification_interface = networks.NetworkInterface(
            self._network, [self.INPUT_NAME, self.TARGET_NAME], [self.OUT_NAME]
        )

    def _assemble_network(self, network_assembler: DenseFeedForwardAssembler):

        target_size = torch.Size([-1])
        network_assembler = network_assembler.set_names(
            input_=self.INPUT_NAME, output=self.OUT_NAME
        )
        constructor = networks.NetworkConstructor(network_assembler.build())
        in_, out = constructor[[self.INPUT_NAME, self.OUT_NAME]].ports
        target = constructor.add_tensor_input(self.TARGET_NAME, out.size, labels=["target"])

        out, = constructor.add_op("Flatten Out", builders.FeedForwardBuilder().flatten(), out, labels=["classifier"])
        loss_assembler: TargetAssembler = TargetAssembler(out.size, target_size).set_names(
            input_=self.OUT_NAME, target=self.TARGET_NAME, objective=self.LOSS_NAME
        ).set_objective(TargetAssembler.BUILDER.bce)
        
        validation_assembler: TargetAssembler = TargetAssembler(out.size, target_size).set_names(
            input_=self.OUT_NAME, target=self.TARGET_NAME, objective=self.LOSS_NAME
        ).set_objective(TargetAssembler.BUILDER.binary_val)

        base_network = BaseNetwork(constructor, [out, target])
        loss_assembler.append(base_network)
        validation_assembler.append(base_network)

        network = constructor.net
        network.to(self._device)
        return network

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
            self._network, self.INPUT_NAME, self.TARGET_NAME, self.LOSS_NAME, self.VALIDATION_NAME, self._loss_assembler.loss_names
        )
        
        self._classification_interface = networks.NetworkInterface(
            self._network, [self.INPUT_NAME, self.TARGET_NAME], [self.OUT_NAME]
        )
        self._device = device

    def _assemble_network(self, network_assembler: DenseFeedForwardAssembler):

        # Can probably put this in the base class
        target_size = torch.Size([-1])
        network_assembler = network_assembler.set_names(
            input_=self.INPUT_NAME, output=self.OUT_NAME
        )
        constructor = networks.NetworkConstructor(network_assembler.build())
        in_, out = constructor[[self.INPUT_NAME, self.OUT_NAME]].ports
        target = constructor.add_tensor_input(self.TARGET_NAME, out.size, labels=["target"])

        loss_assembler: TargetAssembler = TargetAssembler(out.size, target_size).set_names(
            input_=self.OUT_NAME, target=self.TARGET_NAME, objective=self.LOSS_NAME
        ).set_objective(TargetAssembler.BUILDER.cross_entropy)
        
        validation_assembler: TargetAssembler = TargetAssembler(out.size, target_size).set_names(
            input_=self.OUT_NAME, target=self.TARGET_NAME, objective=self.LOSS_NAME
        ).set_objective(TargetAssembler.BUILDER.multiclass_val)

        base_network = BaseNetwork(constructor, [out, target])
        loss_assembler.append(base_network)
        validation_assembler.append(base_network)

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

        network_assembler = network_assembler.set_names(
            input_=self.INPUT_NAME, output=self.OUT_NAME
        )
        constructor = networks.NetworkConstructor(network_assembler.build())
        in_, out = constructor[[self.INPUT_NAME, self.OUT_NAME]].ports
        target = constructor.add_tensor_input(self.TARGET_NAME, out.size, labels=["target"])

        loss_assembler: TargetAssembler = TargetAssembler(out.size, out.size).set_names(
            input_=self.OUT_NAME, target=self.TARGET_NAME, objective=self.LOSS_NAME
        ).set_objective(TargetAssembler.BUILDER.mse)
        
        validation_assembler: TargetAssembler = TargetAssembler(out.size, out.size).set_names(
            input_=self.OUT_NAME, target=self.TARGET_NAME, objective=self.LOSS_NAME
        ).set_objective(TargetAssembler.BUILDER.mse)

        base_network = BaseNetwork(constructor, [out, target])
        loss_assembler.append(base_network)
        validation_assembler.append(base_network)

        network = constructor.net
        network.to(self._device)
        return network

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


        # # self._validation_assembler = TargetAssembler(out_size, out_size).set_objective(
        # #     TargetAssembler.BUILDER.mse  
        # # )
        # # .set_names(
        # #    input=self.OUT_NAME, target=self.TARGET_NAME, objective=self.VALIDATION_NAME
        # #)

        # self._loss_assembler = TargetAssembler(out_size, out_size).set_objective(
        #     TargetAssembler.BUILDER.mse
        # )
        # # .set_names(
        # #    input=self.OUT_NAME, target=self.TARGET_NAME, objective=self.VALIDATION_NAME
        # # )



        # self._training_assembler = TrainingNetworkAssembler(
        #     network_assembler,
        #     loss_assemblers=[WeightedObjective(self._loss_assembler, "MSE", 1.0)],
        #     validation_assemblers=[WeightedObjective(self._validation_assembler, self.VALIDATION_NAME)]
        # ).set_names(
        #     input=self.INPUT_NAME,
        #     loss=self.LOSS_NAME,
        #     output=self.OUT_NAME,
        #     target=self.TARGET_NAME
        # )


        
        
        
        # ).set_names()


        # self._out_name = 'y'
        # self._target_name = 't'
        # self._input_name = 'x'
        # self._loss_name = 'loss'
        # self._validation_name = 'validation'




        # self._network_assembler = network_assembler
        # loss_builder: ObjectiveBuilder = ObjectiveBuilder()
        # self._loss_builder = loss_builder
        # self._n_out = n_out

        # network_assembler.input_name = "x"
        # network_assembler.output_name = "y"
         
        # network = network_assembler.build()
        # constructor = networks.NetworkConstructor(network)

        # in_, out, t = network[["x", "y", "t"]].ports
        # # TODO: Set the input size
        # self._loss_assembler = TargetObjectiveAssembler().set_objective(TargetObjectiveAssembler.BUILDER.mse)
        # self._loss_assembler.objective_name = "MSE"
        # self._loss_assembler.target_name = "t"
        
        # self._validation_assembler = TargetObjectiveAssembler(
        #     out.size, t.size
        # ).set_objective(TargetObjectiveAssembler.BUILDER.mse)
        # self._validation_assembler.objective_name = "Validation"
        # self._validation_assembler.target_name = "t"
        # self._validation_assembler.input_name = "x"
        
        # self._loss_assembler.set_input_name("x").set_output_name(self._out_name).set_loss_name(
        #     self._loss_name
        # ).set_validation_name(
        #     self._validation_name
        # ).set_target_name(self._target_name)
        
        # self._loss_assembler.append(BaseNetwork(constructor, [out, t]))
        # self._validation_assembler.append(BaseNetwork(constructor, [out, t]))

        # self._network.to(device)
        
        # self._regression_interface = networks.NetworkInterface(
        #     self._network, self._network.get_ports(self._validation_name), by=[self._input_name, self._target_name]
        # )
        # self._device = device