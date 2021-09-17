from . import base
from . import reducers
from . import losses
import torch.nn as nn
import typing


class ObjectiveBuilder(object):

    def build(self):
        raise NotImplementedError


class BaseObjectiveBuilder(ObjectiveBuilder):

    def __init__(
        self, name: str, objective_cls: base.Objective, 
        reduction_cls: reducers.ObjectiveReduction, w: float=1.0
    ):
        self._objective_cls = objective_cls
        self._name = name
        self._reduction_cls = reduction_cls
        self._w = w
    
    def build(self):

        return self._objective_cls(
            self._name, self._reduction_cls, self._w
        )


class NNLossBuilder(ObjectiveBuilder):

    def __init__(self, name: str, loss_cls: nn.Module, loss_args: typing.Dict=None, w: float=1.0):
        self._loss_cls = loss_cls
        self._name = name
        self._w = w
        self._loss_args = loss_args or {}
    
    def build(self):

        return losses.NNLoss(
            self._name, self._loss_cls(**self._loss_args), self._w
        )
