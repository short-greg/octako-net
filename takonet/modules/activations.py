import torch.nn as nn
import torch


class NullActivation(nn.Module):

    def forward(self, x):
        return x


class Scaler(nn.Module):

    def __init__(
        self, lower: float=0., 
        upper: float=1., 
        n_update_limit=None,
    ):
        super().__init__()
        assert lower < upper
        self._lower = lower
        self._upper = upper
        self._n_update_limit = n_update_limit
        self._n_updates = 0
        self._lower_vec = None
        self._upper_vec = None
        self._vecs_set = False
    
    def _update_vecs(self, x):
        if self._vecs_set is False:
            self._lower_vec = torch.min(x, dim=0, keepdim=True)[0]
            self._upper_vec = torch.max(x, dim=0, keepdim=True)[0]
            self._vecs_set = True
        else:
            self._lower_vec = torch.min(
                torch.min(x, dim=0, keepdim=True)[0],
                self._lower_vec
            )
            self._upper_vec = torch.max(
                torch.max(x, dim=0, keepdim=True)[0],
                self._lower_vec
            )
    
    def forward(self, x):
        if self.training and (
            self._n_update_limit is None 
            or self._n_updates < self._n_update_limit
        ):
            self._update_vecs(x)
        elif not self.training and not self._vecs_set:
            return x

        # scale to between 0 and 1. add 1e-5 to ensure there is no / 0
        scaled_x = (x - self._lower_vec) / (self._upper_vec - self._lower_vec + 1e-5) 
        
        if self._lower != 0. or self._upper != 1.:
            return scaled_x * (self._upper - self._lower) + self._lower

        return scaled_x
