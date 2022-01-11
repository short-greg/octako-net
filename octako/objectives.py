import torch
import torch.nn as nn
import typing


class ObjectiveReduction(nn.Module):

    @staticmethod
    def as_str():
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def get_out_size(in_size: torch.Size):
        raise NotImplementedError


class MeanReduction(ObjectiveReduction):
    
    @staticmethod
    def as_str():
        return 'mean'

    def forward(self, x:torch.Tensor):
        return torch.mean(x)
    
    @staticmethod
    def get_out_size(in_size: torch.Size):
        return torch.Size([])


class NullReduction(ObjectiveReduction):

    @staticmethod
    def as_str():
        return 'none'

    def forward(self, x:torch.Tensor):
        return x

    @staticmethod
    def get_out_size(in_size: torch.Size):
        return in_size


class BatchMeanReduction(ObjectiveReduction):
    
    @staticmethod
    def as_str():
        return 'batchmean'

    def forward(self, x: torch.Tensor):
        return x.mean(dim=0).sum()

    @staticmethod
    def get_out_size(in_size: torch.Size):
        return torch.Size([])


class SumReduction(ObjectiveReduction):
    
    @staticmethod
    def as_str():
        return 'sum'

    def forward(self, x: torch.Tensor):
        return x.mean(dim=0).sum()

    @staticmethod
    def get_out_size(in_size: torch.Size):
        return torch.Size([])


class Objective(nn.Module):

    def __init__(
        self,
        reduction_cls: ObjectiveReduction=MeanReduction,
        w: float=1.0, 
    ):
        """[summary]

        Args:
            reduction_cls (ObjectiveReduction): [description]
        """
        super().__init__()
        self._reduction = reduction_cls()
        self._w = w


class Fitness(Objective):

    MAXIMIZE = True

    def __init__(
        self, reduction_cls: typing.Type[ObjectiveReduction]=MeanReduction, w=1.0
    ):

        super().__init__(
            reduction_cls=reduction_cls, w=w
        )

    def forward(self, x, t):
        raise NotImplementedError


class ClassificationFitness(Fitness):
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        dim = t.dim()
        return self._w * self._reduction(
            (torch.argmax(x, dim=dim) == t).float() 
        ).detach()


class BinaryClassificationFitness(Fitness):

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        
        # classification = self.classify(x).view(-1)

        return self._w * self._reduction(
            (x.round() == t).float() 
        ).detach()


class Loss(Objective):

    def __init__(
        self, 
        reduction_cls: ObjectiveReduction=MeanReduction,
        w: float=1.0, 
    ):
        """[summary]

        Args:
            reduction_cls (ObjectiveReduction): [description]
        """
        super().__init__(reduction_cls, w)

    def forward(self, x, t):

        raise NotImplementedError


class GaussianMinLoss(Loss):

    def __init__(
        self, s: float=1.0, reduction_cls: ObjectiveReduction=MeanReduction,
        w: float=1.0, eps:float=1e-5
        
    ):
        """[Loss function to minimize the likelihood function]

        Args:å
            s (float): [The recipricol of weight (standard deviation) of the Gaussian likelihood to minimize]
            eps (float, optional): [Epsilon to avoid 0 divisions]. Defaults to 1e-5.
        """
        super().__init__(reduction_cls, w)
        self.reduction = reduction_cls()
        self.s = s
        self.eps = eps
    
    @property
    def s(self):
        return self._s
    
    @s.setter
    def s(self, s: float):
        assert s > 0
        self._s = s

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, eps: float):
        assert eps >= 0
        self._eps = eps
    
    def forward(self, x, t):
        """[summary]

        Args:
            x ([type]): [description]
            t ([type]): [description]

        Returns:
            [type]: [description]
        """

        numerator = -0.5 * (x - t) ** 2 + self._eps

        if self._s != 1:
            denominator = self._s ** 2
            exponential = numerator / denominator
        else:
            exponential = numerator

        return self.reduction(-torch.log(
            1 - torch.exp(exponential)
        ))


class NNLoss(Loss):
    """[Wrap a loss object in an Eval]

    """

    def __init__(self, loss: nn.Module, w: float=1.0):

        # It is a loss so maximization is false
        super().__init__(NullReduction, w)
        self.loss = loss

    def forward(self, x, t):
        return self._w * self.loss(x, t)
        

# class MSERecipricolLoss(nn.Module):

#     def __init__(
#         self, weight=1.0, reduction='mean'
#     ):
#         """[summary]

#         Args:
#             weight (float, optional): [description]. Defaults to 1.0.
#             reduction (str, optional): [description]. Defaults to 'mean'.
#         """
#         self._reduction = reduction
#         self._weight = weight
#         self._eps = 1e-5
    
#     def forward(self, x, t):

#         y = (x - t) ** 2 + self._eps
#         y = self._weight * 1 / y
#         if self._reduction == 'None':
#             return y
#         if self._reduction == 'mean':
#             return torch.mean(y)
#         if self._reduction == 'batchmean':
#             return torch.mean(y, dim=0)
#         return y


class PreprocessLoss(nn.Module):

    def __init__(
        self, preprocess_x: nn.Module, preprocess_y: nn.Module, 
        loss_f: nn.Module
    ):

        super().__init__()
        self.preprocess_x = preprocess_x
        self.preprocess_y = preprocess_y
        self.loss_f = loss_f
    
    def forward(self, x, t):

        x = self.preprocess_x(x)
        t = self.preprocess_y(t)
        return self.loss_f(x, t)


class NegLoss(nn.Module):

    def __init__(
        self, loss
    ):
        super().__init__()
        self.loss = loss
    
    def forward(self, x, t):

        return -self.loss(x, t)


class Regularizer(Objective):

    def __init__(
        self, reduction_cls: ObjectiveReduction=MeanReduction,
        w: float=1.0, 
    ):
        """[summary]

        Args:
            reduction_cls (ObjectiveReduction): [description]
        """
        super().__init__(reduction_cls, w)

    def forward(self, x):

        raise NotImplementedError


# move to regularizers
class LogL2Reg(Regularizer):

    def forward(self, x):
        x = torch.clamp(x, -1e-5, 1e5)
        return self._w * self._reduction(torch.log(x) ** 2)


# move to regularizers
class L2Reg(Regularizer):

    def forward(self, x):
        x = torch.clamp(x, -1e-5, 1e5)
        return self._w * self._reduction(x ** 2)


class LogL1Reg(Regularizer):

    def forward(self, x):
        x = torch.clamp(x, -1e-5, 1e5)
        return self._w * self._reduction(torch.abs(torch.log(x)))
    

class L1Reg(Regularizer):

    def forward(self, x):
        x = torch.clamp(x, -1e-5, 1e5)
        return self._w * self._reduction(torch.abs(x))


class GaussianKLDivReg(nn.Module):
    # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes

    def forward(
        self, theta: torch.FloatTensor
    ):
        """[summary]

        Args:
            mu (torch.FloatTensor): [description]
            log_var (torch.FloatTensor): [description]

        Returns:
            [type]: [description]
        """
        p_size = theta.size(1) // 2
        mu, log_var = theta[:, :p_size], theta[:,p_size:]
        # log_var: torch.FloatTensor
        mu = mu.view(mu.size(0), -1)
        log_var = log_var.view(log_var.size(0), -1)
        result = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp(), dim=1)
        # result = 0.5 * (-torch.sum(log_var + 1, dim=1) + torch.sum(torch.exp(log_var), dim=1) + torch.sum(mu ** 2, dim=1))
        result = result.view(-1, 1)
        return self._w * self._reducer(result)

# TODO: Check if still used
class CompoundLoss(nn.Module):

    def __init__(self, weights: list):
        super().__init__()
        self._weights = weights

    def forward(self, xs: typing.List[torch.Tensor]):
        weighted = [x * w for x, w in zip(xs, self._weights)]
        return sum(weighted)


