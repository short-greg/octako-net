# from .utils import StudyItem
# import torch.utils.data as data_utils
# from functools import partial
# from .materials import Material
# import typing


# class Curriculum(object):
#     '''
    
#     Right now only adding indirection.. In the future
#     '''

#     def __init__(self, training=None, validation=None, testing=None):

#         self._training = training
#         self._validation = validation
#         self._testing = testing

    
#     @property
#     def n_training_iterations(self):
#         raise NotImplementedError

#     @property
#     def n_testing_iterations(self):
#         raise NotImplementedError

#     @property
#     def n_validation_iterations(self):
#         raise NotImplementedError
    
#     @property
#     def n_rounds(self):
#         raise NotImplementedError

#     def init_testing_progress(self):

#         raise NotImplementedError
    
#     def init_validation_progress(self):

#         raise NotImplementedError

#     def init_training_progress(self):

#         raise NotImplementedError

#     def iterate_training(self) -> StudyItem:
#         raise NotImplementedError

#     def iterate_testing(self) -> StudyItem:

#         raise NotImplementedError

#     def iterate_testing(self) -> StudyItem:
#         raise NotImplementedError


# class StandardCurriculum(Curriculum):

#     def __init__(self, training=None, validation=None, testing=None, batch_size=32, n_rounds=1, mapper: list=None):

#         super().__init__(training, validation, testing)
#         assert batch_size > 1
#         assert n_rounds >= 1
#         self._batch_size = batch_size
#         self._n_rounds = n_rounds
#         self._mapper = mapper

#         self.iterate_training = partial(self._iterator, dataset=self._training, to_shuffle=True)
#         self.iterate_validation = partial(self._iterator, dataset=self._validation, to_shuffle=False)
#         self.iterate_testing = partial(self._iterator, dataset=self._testing, to_shuffle=False)
    
#     def _map(self, values: list):

#         if self._mapper is not None:
#             return {
#                 k: values[i] for i, k in enumerate(self._mapper)
#             }
        
#         return dict(enumerate(values))
    
#     @property
#     def n_rounds(self):
#         return self._n_rounds
    
#     @n_rounds.setter
#     def n_rounds(self, n_rounds):
#         assert n_rounds >= 1
#         self._n_rounds = n_rounds

#     @property
#     def batch_size(self):
#         return self._batch_size

#     @batch_size.setter
#     def batch_size(self, batch_size):
#         assert batch_size > 1
#         self._batch_size = batch_size

#     @property
#     def n_training_iterations(self):
#         return len(self._training) // self._batch_size

#     @property
#     def n_testing_iterations(self):
#         return len(self._testing) // self._batch_size

#     @property
#     def n_validation_iterations(self):
#         return len(self._validation) // self._batch_size
    
#     def _iterator(self, dataset, to_shuffle) -> StudyItem:

#         if dataset is None:
#             return

#         for study_item in data_utils.DataLoader(
#             dataset, self._batch_size, shuffle=to_shuffle
#         ):
#             yield StudyItem(self._map(study_item))


# class CurriculumBuilder(object):

#     def __init__(self, curriculum_cls: Curriculum, params: dict):

#         self._curriculum_cls = curriculum_cls
#         self._params = params
    
#     def build(self, training, validation, testing):

#         return self._curriculum_cls(
#             training=training, validation=validation, testing=testing, **self._params
#         )


# class StandardCurriculumBuilder(object):

#     def __init__(self, n_rounds: int, batch_size: int, mapper: typing.List=None):
#         self._n_rounds = n_rounds
#         self._batch_size = batch_size
#         self._mapper = mapper

    
#     def build(self, material: Material):
#         return StandardCurriculum(
#             material.training, material.validation, material.testing, batch_size=self._batch_size, 
#             n_rounds=self._n_rounds, mapper=self._mapper
#         )


# class TestingCurriculumBuilder(object):

#     def __init__(self, n_rounds, batch_size: int, mapper: typing.List=None):
#         self.n_rounds = n_rounds
#         self._batch_size = batch_size
#         self._mapper = mapper
    
#     def build(self, material):
#         return StandardCurriculum(
#             material.training_for_test, None, material.testing, 
#             batch_size=self._batch_size, n_rounds=self.n_rounds, mapper=self._mapper
#         )


