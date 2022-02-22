"""
Basic learning machine classes. 

A learning machine consists of parameters, operations and the interface
Mixins are used to flexibly define the interface for a learning machine
Each Mixin is a "machine component" which defines an interface for the
user to use. Mixins make it easy to reuse components.

class BinaryClassifierLearner(Learner, Tester, Classifier):
  
  def __init__(self, ...):
      # init operations

  def classify(self, x):
      # classify

  def learn(self, x, t):
      # update parameters of network
    
  def test(self, x, t):
      # evaluate the ntwork

"""

from dataclasses import dataclass
import torch
import torch.optim
import torch.nn as nn
from abc import abstractmethod


def todevice(f):
    """decorator for converting the args to the device of the learner
    """
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return result

    return wrapper


def cpuresult(f):
    """decorator for converting the results to cpu
    """
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return result.cpu()

    return wrapper


def dict_cpuresult(f):
    """decorator for converting the results in dict format to cpu
    """
    def wrapper(self, *args):
        args = tuple(a.to(self._device) for a in args)
        result = f(self, *args)
        return {
            k: t.cpu() for k, t in result.items()
        }
    return wrapper


class MachineComponent(nn.Module):
    """Base class for component. Use to build up a Learning Machine
    """

    def __init__(self, device='cpu'):
        super().to(device)
        self._device = device
    
    def to(self, device):
        super().to(device)
        self._device = device

    def is_(self, component_cls):
        if isinstance(self, component_cls):
            return True


class Learner(MachineComponent):
    """Update the machine parameters
    """

    @abstractmethod
    def learn(self, x, t):
        """Function for learning the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError


class Tester(MachineComponent):
    """Evaluate the machine
    """

    @abstractmethod
    def test(self, x, t):
        """Function for evaluating the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        raise NotImplementedError


class Regressor(MachineComponent):
    """Output a real value
    """

    @abstractmethod
    def regress(self, x: torch.Tensor):
        raise NotImplementedError


class Classifier(MachineComponent):
    """Output a categorical value
    """

    @abstractmethod
    def classify(self, x: torch.Tensor):
        raise NotImplementedError
