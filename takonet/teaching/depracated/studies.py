
# from abc import ABC, abstractmethod
# import typing
# from . import Learner
# from .data_controllers import ExperimentAccessor, DataController


# class StudyBuilder(ABC):

#     optimize_field: str = None
#     optimize_module: str = None

#     @abstractmethod
#     def build(self, name: str) -> typing.Tuple[
#         Learner, ExperimentAccessor
#     ]:
#         raise NotImplementedError


# class Study(object):

#     def run(name):
#         raise NotImplementedError


# class SingleStudy(Study):

#     def __init__(self, study_builder: StudyBuilder):
#         self.study_builder = study_builder

#     def run(self, name):
        
#         data_controller = DataController()
#         # create the study accessor
#         # experiment_accessor = data_controller.create_experiment(name)
#         study_accessor = data_controller.create_study("Single")
        
#         # TODO: Upgrade this
        
#         # learner, dojo = self.study_builder.build_base("Single Experiment", study_accessor)
#         # dojo.run(learner)
