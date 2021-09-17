import torch.nn as nn
import torch


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
