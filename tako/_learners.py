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
    
  def forward(self, x):
      # standard forward method

"""

from dataclasses import dataclass
import typing
import torch
import torch.optim
import torch.nn as nn
from abc import ABC, abstractmethod
from ._networks import Port, Network


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
        print(list(result.values()))
        return {
            k: t.cpu() for k, t in result.items()
        }
    return wrapper


@dataclass
class Attributes(ABC):
    pass


@dataclass
class LearnerAttributes(object):

    net: Network
    optim: torch.optim.Optimizer
    in_: Port
    out: Port
    x: Port
    loss: Port
    validation: Port
    t: Port


class MachineComponent(nn.Module):
    """Base class for component. Use to build up a Learning Machine
    """

    def __init__(self, device='cpu'):
        super().__init__()
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


class LearningMachine(Learner, Tester):
    """Base class for a learning machine
    """

    def __init__(self, device='cpu'):
        super().__init__(device)
        self._p = self.build()

    @abstractmethod
    def build(self) -> LearnerAttributes:
        raise NotImplementedError
    
    @todevice
    @dict_cpuresult
    def learn(self, x: torch.Tensor, t: torch.Tensor):
        self._p.optim.zero_grad()
        loss, validation = self._p.net.probe(
            [self._p.loss, self._p.validation], {self._p.x.node: x, self._p.t.node: t}
        )
        loss.backward()
        self._p.optim.step()
        return {
            'Loss': loss,
            'Validation': validation
        }

    @todevice
    @dict_cpuresult
    def test(self, x: torch.Tensor, t: torch.Tensor):
        loss, validation = self._p.net.probe([self._p.loss, self._p.validation], {self._p.x.node: x, self._p.t.node: t})

        return {
            'Loss': loss,
            'Validation': validation
        }

    @todevice
    @cpuresult
    def forward(self, x: torch.Tensor):
        return self._p.net.probe(self._p.out, by={self._p.x.node: x})
