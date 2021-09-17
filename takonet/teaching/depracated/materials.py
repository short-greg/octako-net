# from enum import Enum
# import torch


# class DatasetType(Enum):

#     TRAINING = "training"
#     TESTING = "testing"
#     VALIDATION = "validation"


# class Material(object):

#     def __init__(self):

#         self._training = None
#         self._testing = None
#         self._validation = None
#         self._training_for_test = None

#     @property
#     def fields(self):
#         raise NotImplementedError

#     @property
#     def training(self):

#         if self._training:
#             return self._training
        
#         self._training, self._validation = self.load_training_and_validation()

#         return self._training

#     @property
#     def testing(self):

#         if self._testing:
#             return self._testing
        
#         self._testing = self.load_testing()
#         return self._testing

#     @property
#     def validation(self):

#         if self._validation:
#             return self._validation
        
#         self._training, self._validation = self.load_training_and_validation()
#         return self._validation
    
#     @property
#     def training_for_test(self):

#         if self._training_for_test:
#             return self._training_for_test
        
#         self._training_for_test = self.load_training_for_test()
#         return self._training_for_test
    
#     def load_training_and_validation(self):

#         raise NotImplementedError()

#     def load_testing(self):

#         raise NotImplementedError()

#     def load_training_for_test(self):
#         '''

#         '''
#         return torch.utils.data.ConcatDataset(
#             [self.training, self.validation]
#         )


# class DatasetSubset(torch.utils.data.Dataset):

#     def __init__(
#         self, dataset: torch.utils.data.Dataset, 
#         indices: torch.LongTensor
#     ):

#         self._dataset = dataset
#         self._indices = indices
    
#     def __getitem__(self, idx):
#         return self._dataset[self._indices[idx]]
    
#     def __len__(self):
#         return len(self._indices)


# def split_dataset_in_two(dataset_size, percent_first):
#     perm = torch.randperm(dataset_size)
#     end_idx = int(len(dataset_size) * percent_first)

#     return perm[:end_idx], perm[end_idx:]

