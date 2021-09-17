from enum import Enum
from os import stat
from takonet.modules.activations import NullActivation, Scaler
from torch import nn
import torch
from takonet.modules import objectives
from .networks import Operation
import typing


"""
Overview: Modules for building layers in a network

They are meant to be extensible so that a builder can be replaced as long as it 
provides the same interface. 

They provide explicit functions like "relu" and also non-explicit functions
like "activation". The user can set activation to be any type of activation
as long as the interface is correct.
"""


class FeedForwardBuilder(object):
    """Builder for creating modules that compose a feed forward network
    """

    def __init__(self):
        self._activation: typing.Callable[[], Operation] = self.relu
        self._normalizer: typing.Callable[[int], Operation] = self.batch_normalizer
        self._dense: typing.Callable[[int, int], Operation] = self.linear
        self._out_activation: typing.Callable[[], Operation] = self.sigmoid

        self.ACTIVATION_MAP = dict(
            sigmoid=self.sigmoid,
            relu=self.relu,
            tanh=self.tanh,
            null=self.null
        )

    def activation_type(self, activation_type: str):
        return self.ACTIVATION_MAP[activation_type]
    
    def relu(self, in_size: torch.Size) -> Operation:
        return Operation(nn.ReLU(), in_size)
    
    def sigmoid(self, in_size: torch.Size) -> Operation:
        return Operation(nn.Sigmoid(), in_size)

    def tanh(self, in_size: torch.Size) -> Operation:
        return Operation(nn.Tanh(), in_size)
    
    def null(self, in_size: torch.Size) -> Operation:
        return Operation(NullActivation, in_size)

    def batch_normalizer(self, in_features: int, eps: float=1e-4, momentum: float=0.1) -> Operation:
        return Operation(nn.BatchNorm1d(in_features, eps, momentum), torch.Size([-1, in_features]))

    def instance_normalizer(self, in_features: int, eps: float=1e-4, momentum: float=0.1) -> Operation:
        return Operation(nn.InstanceNorm1d(in_features, eps, momentum), torch.Size([-1, in_features]))

    def dropout(self, in_size: torch.Size, p: float=None):
        p = p or self._p
        return Operation(nn.Dropout(p), in_size)

    def linear(self, in_features: int, out_features: int, bias: bool=True):
        return Operation(nn.Linear(
            in_features, out_features, 
        ), torch.Size([-1, out_features]))


class ReductionType(Enum):

    MeanReduction = objectives.reducers.MeanReduction
    SumReduction = objectives.reducers.SumReduction
    NullReduction = objectives.reducers.NullReduction
    BatchMeanReduction = objectives.reducers.BatchMeanReduction


class LossBuilder(object):
    """Builder for creating loss function modules
    """

    def __init__(self):
        self._regularizer: typing.Callable[[int, float]] = self.l2_reg
        self._loss: typing.Callable[[int, float]] = self.mse
        self._validator: typing.Callable[[int, float], Operation] = self.binary_classifier
        self._target_processor: typing.Callable[[torch.Size], Operation] = self.null_processor

    def mse(self, in_size: torch.Size, weight: typing.Union[float, torch.Tensor]=1.0, reducer: objectives.ObjectiveReduction=objectives.MeanReduction):

        # TODO: Add in a weight
        return Operation(
            nn.MSELoss(reduction=reducer.as_str()), reducer.get_out_size(in_size)
        )

    def bce(self, in_size: torch.Size, weight: typing.Union[float, torch.Tensor]=1.0, reducer: objectives.ObjectiveReduction=objectives.MeanReduction):

        return Operation(
            nn.BCELoss(reduction=reducer.as_str()), reducer.get_out_size(in_size)
        )

    def cross_entropy(self, in_size: torch.Size, weight: typing.Union[float, torch.Tensor]=1.0, reducer: objectives.ObjectiveReduction=objectives.MeanReduction):

        return Operation(
            nn.CrossEntropyLoss(reduction=reducer.as_str()), reducer.get_out_size(in_size)
        )

    def l1_reg(self, in_size: torch.Size, weight: typing.Union[float, torch.Tensor], reducer: objectives.ObjectiveReduction=objectives.MeanReduction):

        return Operation(
            objectives.L1Reg(reduction=reducer, weight=weight), reducer.get_out_size(in_size)
        )

    def l2_reg(self, in_size: torch.Size, weight: typing.Union[float, torch.Tensor]=1.0, reducer: objectives.ObjectiveReduction=objectives.MeanReduction):

        return Operation(
            objectives.L2Reg(reduction=reducer, weight=weight), reducer.get_out_size(in_size)
        )
    
    def binary_classifier(self, in_size: torch.Size, reducer: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction):

        return Operation(
            objectives.BinaryClassificationFitness(reduction_cls=reducer), reducer.get_out_size(in_size)
        )

    def multiclassifier(self, in_size: torch.Size, reducer: typing.Type[objectives.ObjectiveReduction]=objectives.MeanReduction):

        return Operation(
            objectives.ClassificationFitness(reduction_cls=reducer), reducer.get_out_size(in_size)
        )

    def scale(self, in_size: torch.Size):
        """Whether to scale the target between 0 and 1
        """
        return Operation(Scaler(), in_size)
    
    def null_processor(self, in_size: torch.Size):
        return Operation(NullActivation(), in_size)
