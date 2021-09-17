import numpy as np
import torch.nn as nn
from src import core
import torch
import typing


class OptimizerGroup(object):

    def __init__(self, optimizer_dict):

        super().__init__()
        self._optimizer_dict = optimizer_dict
    
    def state_dict(self):

        return {
            k: opt.state_dict() for k, opt in self._optimizer_dict.items()
        }

    def load_state_dict(self, state_dict):

        for k, v in state_dict.items():
            self._optimizer_dict[k].load_state_dict(v)


class Learner(object):

    def __init__(
        self, network: nn.Module, 
        loss_calculator: nn.Module,
        optimizer_group: OptimizerGroup, 
        device='cpu'
    ):
        '''
        '''
        self.device = device
        self._network = network
        self._optimizer_group = optimizer_group
        self._loss_calculator = loss_calculator
        self._training = None

    @property
    def aggregate_fields(self):
        return self._loss_calculator.aggregate_fields

    @property
    def aggregate_fields(self):
        return self._loss_calculator.evaluation_fields

    def state_dict(self):
        return dict(
            network=self._network.state_dict(),
            optimizer_group=self._optimizer_group.state_dict(),
            device=self.device,
        )

    def load_state_dict(self, state_dict):
        self._network.load_state_dict(state_dict['network'])
        self._optimizer_group.load_state_dict(state_dict['optimizer_group'])
        self.to(state_dict['device'])
    
    @property
    def validation_target(self):
        return self._loss_calculator.validator.name

    @property
    def validation_maximize(self):
        return self._loss_calculator.validator.maximize

    def set_state(self, training=True):

        if (self._training is None or self._training is False) and training is True:
            self._training = True
            self._network.train()
        if (self._training is None or self._training is True) and training is False:
            self._training = False
            self._network.eval()

    def reset(self):
        raise NotImplementedError

    def to(self, device):
        self.device = device
        self.network.to(device)

    def test(self, study_item: core.study_item.StudyItem) -> dict:
        raise NotImplementedError

    def learn(self, study_item: core.study_item.StudyItem) -> dict:
        raise NotImplementedError


class BinaryClassifier(object):

    def execute(self, x: torch.Tensor): 
        raise NotImplementedError

    def classify(self, x: torch.Tensor):
        raise NotImplementedError


class Regressor(object):

    def execute(self, x: torch.Tensor): 
        raise NotImplementedError

    def regress(self, x):
        return self.execute(x)


class LossCalculator(nn.Module):

    @property
    def fields(self):
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, record_evaluation: bool=False) -> typing.Tuple[dict, dict]:
        raise NotImplementedError





# class LearnerDecorator(Learner):

#     def __init__(self, base_learner):
#         '''
#         Args
#             Used for adding more functionality to a learner
#         '''
#         super().__init__(
#             base_learner._network, 
#             base_learner._loss_calculator,
#             base_learner._optimizer_group, 
#             base_learner.device
#         )

#         self._base_learner = base_learner

#         self.state_dict = base_learner.state_dict
#         self.load_state_dict = base_learner.load_state_dict
#         self.set_state = base_learner.set_state
#         self.to = base_learner.to
#         self.test = base_learner.test
#         self.learn = base_learner.learn
    
#     def __getattr__(self, key):

#         return self._base_learner.__getattribute__(key)


# class ClassifierLearner(LearnerDecorator):

#     def __init__(self, base_learner, classify_helper):
#         '''
#         Args
#             base_learner Learner - The learner to learn on
#             classify - classify()
#         '''

#         super().__init__(base_learner)
#         self._classify_helper = classify_helper
    
#     def classify(self, x):
#         '''
#         Args:

#         '''
#         y = self._base_learner.net(x)

#         return self._classify_helper(y)


# class RegressorLearner(LearnerDecorator):

#     def __init__(self, base_learner):
#         '''
#         Args
#             base_learner Learner - The learner to learn on
#             classify - classify()
#         '''

#         super().__init__(base_learner)
    
#     def regress(self, x):
#         '''
#         Args:

#         '''
#         return self._base_learner.net(x)


# class StudyItem(object):

#     def __init__(self, item_dict: dict(str=torch.Tensor)):
#         """[summary]

#         Args:
#             item_dict (dict): [description]

#         Returns:
#             [type]: [description]
#         """
#         self._item_dict = item_dict
    
#     def __getattr__(self, k):
#         """[summary]

#         Args:
#             k ([type]): [description]

#         Raises:
#             AttributeError: [The value k is not in the study item dict]

#         Returns:
#             [torch.Tensor]: [The value for the study item]
#         """

#         val = self._item_dict.get(k)
#         if val is None:
#             raise AttributeError
        
#         return val
