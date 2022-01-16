"""
Basic learning machine classes. 

Learning Machines define an operation, a learning algorithm
for learning that operation and a testing algorithm for testing the operation.
"""

from dataclasses import dataclass
import torch
from . import networks
import torch.optim
import torch.nn
from abc import abstractmethod


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


class LearnerMixin(object):

    def check_interface(self, net: networks.Network, in_, out):
        if in_ not in net:
            raise ValueError(f"Inputs {in_} not in the network.")
        
        if out not in net:
            raise ValueError(f'Outputs {out} not in the network.')

    def __init__(self, net):
        """initializer for learner. Will only be executed once

        Args:
            net : Network used by learner. Can be a "Network" or an
            object composed of multiple networks
        """
        self._net = net


class Learner(LearnerMixin):
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


class Validator(LearnerMixin):
    """Base learning machine class
    """
    @abstractmethod
    def test(self, x, t):
        """Function for evaluating the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError


class Regressor(LearnerMixin):

    @abstractmethod
    def regress(self, x, t):
        raise NotImplementedError


class SimpleRegressor(Regressor):

    def __init__(self, net):
        super().__init__(net)
        self.check_interface(self, 'x', 'y')
    
    def regress(self, x: torch.Tensor):
        return self._net.probe('y', by={x: x})['y']


class Classifier(LearnerMixin):

    @abstractmethod
    def classify(self, x: torch.Tensor):
        raise NotImplementedError


class BinaryClassifier(Classifier):

    def __init__(self, net):
        super().__init__(net)
        self.check_interface(self, 'x', 'y')

    def classify(self, x: torch.Tensor):
        y = self._net.probe('y', by={x: x})['y']
        return (y >= 0.5).float()
