# from src.core.study_item import StudyItem
# import pytest
# import torch.utils.data
# from torch.utils.data.dataset import TensorDataset
# from . import curriculums


# class TestStandardCurriculum(object):

#     TRAINING_SAMPLES = 200
#     VALIDATION_SAMPLES = 40
#     TESTING_SAMPLES = 80

#     BASE_N_ROUNDS = 10

#     @property
#     def datasets(self):
#         training = torch.randn(self.TRAINING_SAMPLES, 2)
#         validation = torch.randn(self.VALIDATION_SAMPLES, 2)
#         testing = torch.randn(self.TESTING_SAMPLES, 2)

#         return (
#             torch.utils.data.TensorDataset(training),
#             torch.utils.data.TensorDataset(validation),
#             torch.utils.data.TensorDataset(testing)
#         )
    
#     def test_batch_size_with_valid(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=16, n_rounds=self.BASE_N_ROUNDS)
#         curriculum.batch_size = 64
#         assert curriculum.batch_size == 64
    
#     def test_batch_size_with_invalid(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=16, n_rounds=self.BASE_N_ROUNDS)
#         with pytest.raises(AssertionError):
#             curriculum.batch_size = -1

#     def test_batch_size_with_valid_in_init(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=16, n_rounds=self.BASE_N_ROUNDS)
#         assert curriculum.batch_size == 16

#     def test_n_rounds_in_init_with_valid(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=16, n_rounds=self.BASE_N_ROUNDS)
#         assert curriculum.n_rounds == 10

#     def test_n_rounds_with_valid(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=16, n_rounds=self.BASE_N_ROUNDS)
#         curriculum.n_rounds = 100
#         assert curriculum.n_rounds == 100

#     def test_batch_size_with_invalid(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=16, n_rounds=self.BASE_N_ROUNDS)
#         with pytest.raises(AssertionError):
#             curriculum.n_rounds = 0

#     def test_n_testing_iterations(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=10, n_rounds=self.BASE_N_ROUNDS)
#         assert curriculum.n_testing_iterations == self.TESTING_SAMPLES // self.BASE_N_ROUNDS
        
#     def test_n_training_iterations(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=10, n_rounds=self.BASE_N_ROUNDS)
#         assert curriculum.n_training_iterations == self.TRAINING_SAMPLES // self.BASE_N_ROUNDS
        
#     def test_n_validation_iterations(self):

#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=10, n_rounds=self.BASE_N_ROUNDS)
#         assert curriculum.n_validation_iterations == self.VALIDATION_SAMPLES // self.BASE_N_ROUNDS
    
#     def test_iterate_training(self):

#         i = 0
#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=10, n_rounds=self.BASE_N_ROUNDS)
#         for i, study_item in enumerate(curriculum.iterate_training()):
#             assert type(study_item) == StudyItem
        
#         assert (i + 1) == curriculum.n_training_iterations
        
#     def test_iterate_validation(self):

#         i = 0
#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=10, n_rounds=self.BASE_N_ROUNDS)
#         for i, study_item in enumerate(curriculum.iterate_validation()):
#             assert type(study_item) == StudyItem
        
#         assert (i + 1) == curriculum.n_validation_iterations

#     def test_iterate_testing(self):

#         i = 0
#         curriculum = curriculums.StandardCurriculum(*self.datasets, batch_size=10, n_rounds=self.BASE_N_ROUNDS)
#         for i, study_item in enumerate(curriculum.iterate_testing()):
#             assert type(study_item) == StudyItem
        
#         assert (i + 1) == curriculum.n_testing_iterations
