"""
Basic learning machine classes. 

Learning Machines define an operation, a learning algorithm
for learning that operation and a testing algorithm for testing the operation.
"""

from dataclasses import dataclass
import typing
import torch
from . import networks
import torch.optim
import torch.nn as nn
from abc import abstractclassmethod, abstractmethod, abstractproperty


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


class MachineComponent(nn.Module):

    op = None

    def __call__(self, *t: torch.Tensor):
        raise NotImplementedError

    def is_(self, component_cls):
        if isinstance(self, component_cls):
            return True


class LearningMachine(nn.Module):

    def __init__(self, components: typing.List[MachineComponent]):
        
        # TODO: Check they do not conflict
        for v in components:
            setattr(self, v.name(), v)
        self._components = components
    
    @property
    def components(self):
        return self._components
    
    def is_(self, component_cls: typing.Type):
        
        for component in self._components:
            if component.is_(component_cls):
                return True
        return False


def is_machine(obj: LearningMachine, cls: typing.Type[MachineComponent]):
    """Check if a learning machine contains component cls"""

    return isinstance(getattr(obj, cls.op), cls)


# TODO: Think a little more about the components
# add in a compound component?


# TODO: Update these components

class Learner(MachineComponent):
    """Base learning machine class
    """
    op = 'learn'

    @abstractmethod
    def learn(self, x, t):
        """Function for learning the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError


class Validator(MachineComponent):
    """Base learning machine class
    """
    op = 'test'

    @abstractmethod
    def test(self, x, t):
        """Function for evaluating the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError


class Regressor(MachineComponent):

    op = 'regress'

    @abstractmethod
    def regress(self, x, t):
        raise NotImplementedError


# class SimpleRegressor(Regressor):

#     def __init__(self, net):
#         super().__init__(net)
#         self.check_interface(self, 'x', 'y')
    
#     def regress(self, x: torch.Tensor):
#         return self._net.probe('y', by={x: x})['y']


class Classifier(MachineComponent):

    op = 'classify'

    @abstractmethod
    def classify(self, x: torch.Tensor):
        raise NotImplementedError


# class BinaryClassifier(Classifier):

#     def __init__(self, net):
#         super().__init__(net)
#         self.check_interface(self, 'x', 'y')

#     def classify(self, x: torch.Tensor):
#         y = self._net.probe('y', by={x: x})['y']
#         return (y >= 0.5).float()
