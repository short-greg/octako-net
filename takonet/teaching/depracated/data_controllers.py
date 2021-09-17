# from abc import ABC, abstractmethod
# import pandas as pd
# from torch.utils.data.dataset import Dataset
# from . import events
# import copy
# from enum import Enum

# from dataclasses import dataclass
# import typing
# from src.machines.learners import Learner


# @dataclass
# class ExtensionParams:
#     pass


# class EventTypes(Enum):

#     @classmethod
#     def has_value(cls, value):
#         return value in cls._value2member_map_



# class Progress(ABC):

#     def as_dict(self) -> dict:
#         return self.__dict__

#     @abstractmethod
#     def clone(self):
#         raise NotImplementedError


# class DojoProgress(object):

#     def __init__(
#         self, n_experiments=0, n_finished=0
#     ):
#         self._n_experiments = n_experiments
#         self._n_finished = n_finished
    
#     @property
#     def n_finished(self):
#         return self._n_finished

#     @property
#     def n_experiments(self):
#         return self._n_experiments
    
#     def start_experiment(self):

#         self.n_experiments += 1

#     def finish_experiment(self):

#         self.n_finished += 1


# class ExperimentProgress(object):

#     def __init__(
#         self, n_base=0, n_base_finished=0, finished=False
#     ):
#         self._n_base = n_base
#         self._n_base_finished = n_base_finished
#         self._finished = finished

#     @property
#     def n_base(self):
#         return self._n_base

#     @property
#     def n_base_finished(self):
#         return self._n_base_finished

#     @property
#     def finished(self):
#         return self._finished
    
#     def start_run(self):

#         self.n_base += 1
    
#     def finish_run(self):

#         self.n_base_finished += 1
    
#     def finish(self):

#         self.finished = True


# class TeacherProgress(object):

#     def __init__(self, n_runs: int, n_finished: int):

#         self._n_runs = n_runs
#         self._n_finished = n_finished
    
#     @property
#     def n_runs(self):
#         return self._n_runs
    
#     @property
#     def n_finished(self):
#         return self._n_finished
    
#     def start(self):

#         self._n_runs += 1
    
#     def finish(self):

#         self._n_finished += 1


# class RunProgress(object):

#     def __init__(
#         self, turn=0, round=0, iteration=0, total_iterations=0, 
#         n_rounds=0, n_iterations=0, finished=False, round_finished=False
#     ):
#         self._round = round
#         self._iteration = iteration
#         self._total_iterations = total_iterations
#         self._n_rounds = n_rounds
#         self._n_iterations = n_iterations
#         self._finished = finished
#         self._turn = turn
#         self._round_finished = round_finished

#     def finish(self):
#         if not self.finished:
#             self.finish_round()
#             self._finished = True
#             return True
#         return False
    
#     def finish_round(self):
#         if self._round_finished:
#             return False
#         self._round_finished = True
#         return True
    
#     def start(self):
#         if not self.finished:
#             return False
            
#         self._turn += 1
#         self._round = 0
#         self._finished = False
#         self._round_finished = False
#         return True
    
#     def update_iteration(self):
#         if self.round_finished or self.finished:
#             return False

#         self._iteration += 1
#         self._total_iterations += 1
#         return True

#     def start_round(self, n_iterations=None):
#         if self.finished or not self.round_finished:
#             return False

#         self._n_iterations = n_iterations or self.n_iterations
#         self._iteration = 0
#         self._round += 1
#         self._round_finished = False
#         return True
    
#     def as_dict(self) -> dict:
#         return self.__dict__

#     def clone(self):

#         return RunProgress(
#             self._turn, self._round, self._iteration, self._total_iterations, 
#             self._n_rounds, self._n_iterations, self._finished, self._round_finished
#         )

#     @property
#     def round(self):
#         return self._round

#     @property
#     def round_finished(self):
#         return self._round_finished

#     @property
#     def total_iterations(self):
#         return self._total_iterations

#     @property
#     def iteration(self):
#         return self._iteration
    
#     @property
#     def n_rounds(self):
#         return self._n_rounds
    
#     @property
#     def n_iterations(self):
#         return self._n_iterations
    
#     @property
#     def finished(self):
#         return self._finished
    
#     @property
#     def turn(self):
#         return self._turn


# class ResultManager(object):

#     TEACHER_COL = 'TeacerID'
#     TURN_COL = 'Turn'
#     ROUND_COL = 'Round'
#     ITER_COL = 'Iteration'

#     BASE_COLUMNS = [TEACHER_COL, TURN_COL, ROUND_COL, ITER_COL]

#     def __init__(self):

#         self._df = pd.DataFrame(columns=self.BASE_COLUMNS)

#     def add_result(self, teacher_id: int, turn: int, round: int, iteration: int, data: dict):

#         columns = [*self.BASE_COLUMNS, *list(data.keys())]
#         values = [teacher_id, turn, round, iteration, *list(data.values())]
        
#         self._df.loc[len(self._df), columns] = values

#     def get(self, retrieve_fields, teacher_ids_filter=None, round_ids_filter=None, iteration_ids_filter=None):

#         df = self._df

#         if teacher_ids_filter is not None:
#             df = df[df[self.TEACHER_COL].isin(teacher_ids_filter)]
#         if round_ids_filter is not None:
#             df = df[df[self.ROUND_COL].isin(round_ids_filter)]
#         if iteration_ids_filter is not None:
#             df = df[df[self.ITER_COL].isin(iteration_ids_filter)]

#         df = df[retrieve_fields]

#         return df



# class RunManager(object):

#     def update_progress(self, id: int, progress: RunProgress):
#         pass

#     def get(self, id: int):
#         pass


# class RunInfo(object):

#     class RunEventTypes(EventTypes):

#         ROUND_STARTED = 'round_started'
#         ROUND_FINISHED = 'round_finished'
#         RESULT_UPDATE = 'result update'
#         STARTED = 'started'
#         FINISHED = 'finished'
#         EXTENSION_UPDATED = 'extension_updated'

#     def __init__(
#         self, run_manager: RunManager, run_id: int, 
#         run_name: str, 
#         extension_params: ExtensionParams, 
#         run_progress: RunProgress=None
#     ):

#         self._id = run_id
#         self._manager = run_manager
#         self._result_manager = run_manager.result_manager
#         self._name = run_name
#         self.result_updated_event = events.TeachingEvent()
#         self.started_event = events.TeachingEvent()
#         self.finished_event = events.TeachingEvent()
#         self.round_started_event = events.TeachingEvent()
#         self.round_finished_event = events.TeachingEvent()
#         self.extension_updated_event = events.TeachingEvent()
#         self.extension_params = extension_params
        
#         if run_progress:
#             self._progress = run_progress.clone()
#         else:
#             self._progress = TeacherProgress()
#             self._manager.update_progress(self._id, self._progress)

#     @property
#     def id(self):
#         return self._id
    
#     @property
#     def name(self):
#         return self._name

#     @property
#     def progress(self) -> RunProgress:
#         return self._progress.clone()

#     def get_event_by_name(self, event_name: str) -> events.TeachingEvent:
#         assert self.RunEventTypes.has_value(event_name) 
#         if event_name == self.RunEventTypes.RESULT_UPDATE.value:
#             return self.result_updated_event
#         elif event_name == self.RunEventTypes.ROUND_STARTED.value:
#             return self.round_started_event
#         elif event_name == self.RunEventTypes.ROUND_FINISHED.value:
#             return self.round_finished_event
#         elif event_name == self.RunEventTypes.FINISHED.value:
#             return self.finished_event
#         elif event_name == self.RunEventTypes.STARTED.value:
#             return self.started_event
#         elif event_name == self.RunEventTypes.EXTENSION_UPDATED.value:
#             return self.extension_updated_event
        
#     # 1) Trigger update events
#     def add_result(self, results: dict):

#         if self._progress.finished:
#             self._progress.start()
#         if self._progress.finish_round:
#             self._progress.start_round()

#         if self._progress.iteration == 0:
#             self.started_event.invoke(self._name)
        
#         self._result_manager.add_result(
#             self._id, self._progress.turn, self._progress.round, 
#             self._progress.iteration, results
#         )
        
#         self._progress.update_iteration()
#         self._manager.update_progress(self._id, self._progress)

#         self.result_updated_event.invoke(self._name)

#     def get_results(self, retrieve_fields, round_ids_filter=None, iteration_ids_filter=None):
        
#         return self._manager.get(
#             retrieve_fields, round_ids_filter=round_ids_filter, 
#             iteration_ids_filter=iteration_ids_filter
#         )

#     def start(self):

#         started = self._progress.start()
#         if started: self._manager.update_progress(self._id, self._progress)
#         self.started_event.invoke(self._name)

#     def finish(self):

#         self._finished = True
#         finished = self._progress.finish()
#         if finished: self._manager.update_progress(self._id, self._progress)
#         self.finished_event.invoke(self._name)

#     # 1) Trigger update events
#     def start_round(self, n_iterations=None):

#         started = self._progress.start_round(n_iterations)
#         if started: self._manager.update_progress(self._id, self._progress)
#         self.round_started_event.invoke(self._name)

#     # 1) Trigger update events
#     def finish_round(self):

#         # self._module_progress.update_round(n_iterations)
#         # self._module_manager.update_progress(self._module_id, self._module_progress)
#         finished = self._progress.finish_round()
#         if finished: self._manager.update_progress(self._id, self._progress)
#         self.round_finished_event.invoke(self._name)


# class TeacherManager(object):

#     EXPERIMENT_COL = 'ExperimentID'
#     STUDY_COL = 'StudyID'
#     NAME_COL = 'Name'
#     RESULT_COL = 'Result Columns'
#     PROGRESS_COL = "Progress"
#     EXTENSION_COL = "Extension Params"
#     TEACHER_ID_COL = 'Teacher ID'


#     BASE_COLUMNS = [TEACHER_ID_COL, STUDY_COL, EXPERIMENT_COL, NAME_COL, EXTENSION_COL, PROGRESS_COL]

#     def __init__(self):

#         self._df = pd.DataFrame(columns=self.BASE_COLUMNS, dtype=object)
#         self._result_manager = ResultManager()

#     def create(self, study_id, experiment_id, name, extension_params: ExtensionParams, progress: TeacherProgress):
#         id = len(self._df)
#         self._df.loc[id, self.BASE_COLUMNS] = [id, study_id, experiment_id, name, extension_params, progress]
#         return id
    
#     def get_teacher_ids(self, name, experiment_id):

#         return list(
#             self._df[
#                 self._df[self.EXPERIMENT_COL] == experiment_id and 
#                 self._df[self.NAME_COL] == name
#             ].index
#         )
         
#     @property
#     def result_manager(self):
#         return self._result_manager
    
#     def update_progress(self, teacher_id: int, teacher_progress: TeacherProgress):
        
#         self._df.loc[teacher_id, self.PROGRESS_COL] = teacher_progress
    
#     def get_teacher(self, experiment_id):

#         return self._df[self._df[self.EXPERIMENT_COL] == experiment_id]

#     def get(self, id):

#         return self._df.iloc[id]


# # The following are the interfaces 
# class TeacherInfo(object):

#     # class EventTypes(Enum):

#     #     ROUND_STARTED = 'round_started'
#     #     ROUND_FINISHED = 'round_finished'
#     #     RESULT_UPDATE = 'result update'
#     #     STARTED = 'started'
#     #     FINISHED = 'finished'
#     #     EXTENSION_UPDATED = 'extension_updated'

#     #     @classmethod
#     #     def has_value(cls, value):
#     #         return value in cls._value2member_map_

#     def __init__(
#         self, teacher_manager: TeacherManager, teacher_id: int, 
#         teacher_name: str, 
#         extension_params: ExtensionParams, 
#         teacher_progress: TeacherProgress=None
#     ):
#         self._id = teacher_id
#         self._teacher_manager = teacher_manager
#         self._result_manager = teacher_manager.result_manager
#         self._name = teacher_name
#         self.result_updated_event = events.TeachingEvent()
#         self.started_event = events.TeachingEvent()
#         self.finished_event = events.TeachingEvent()
#         self.round_started_event = events.TeachingEvent()
#         self.round_finished_event = events.TeachingEvent()
#         self.extension_updated_event = events.TeachingEvent()
#         self.extension_params = extension_params
        
#         if teacher_progress:
#             self._module_progress = teacher_progress.clone()
#         else:
#             self._teacher_progress = TeacherProgress()
#             self._teacher_manager.update_progress(self._id, self._module_progress)
    
#     @property
#     def id(self):
#         return self._id
    
#     @property
#     def name(self):
#         return self._name

#     @property
#     def progress(self) -> TeacherProgress:
#         return self._module_progress.clone()

#     def get_event_by_name(self, event_name: str) -> events.TeachingEvent:
#         pass
#         # assert self.EventTypes.has_value(event_name) 
#         # if event_name == self.EventTypes.RESULT_UPDATE.value:
#         #     return self.result_updated_event
#         # elif event_name == self.EventTypes.ROUND_STARTED.value:
#         #     return self.round_started_event
#         # elif event_name == self.EventTypes.ROUND_FINISHED.value:
#         #     return self.round_finished_event
#         # elif event_name == self.EventTypes.FINISHED.value:
#         #     return self.finished_event
#         # elif event_name == self.EventTypes.STARTED.value:
#         #     return self.started_event
#         # elif event_name == self.EventTypes.EXTENSION_UPDATED.value:
#         #     return self.extension_updated_event
        
#     # 1) Trigger update events
#     # def add_result(self, results: dict):

#     #     if self.progress.finished:
#     #         self.progress.start()
#     #     if self.progress.finish_round:
#     #         self.progress.start_round()

#     #     if self.progress.iteration == 0:
#     #         self.started_event.invoke(self._name)
        
#     #     self._result_manager.add_result(
#     #         self._id, self._teacher_progress.turn, self._teacher_progress.round, 
#     #         self._module_progress.iteration, results
#     #     )
        
#     #     self._teacher_progress.update_iteration()
#     #     self._teacher_manager.update_progress(self._id, self._teacher_progress)

#     #     self.result_updated_event.invoke(self._name)
    
#     # def get_results(self, retrieve_fields, round_ids_filter=None, iteration_ids_filter=None):
        
#     #     return self._result_manager.get(
#     #         retrieve_fields, round_ids_filter=round_ids_filter, 
#     #         iteration_ids_filter=iteration_ids_filter
#     #     )

#     def start(self):

#         started = self._teacher_progress.start()
#         if started: self._teacher_manager.update_progress(self._id, self._teacher_progress)
#         self.started_event.invoke(self._name)

#     def finish(self):

#         self._finished = True
#         finished = self._teacher_progress.finish()
#         if finished: self._teacher_manager.update_progress(self._id, self._teacher_progress)
#         self.finished_event.invoke(self._name)

#     # 1) Trigger update events
#     def start_round(self, n_iterations=None):

#         started = self._teacher_progress.start_round(n_iterations)
#         if started: self._teacher_manager.update_progress(self._id, self._teacher_progress)
#         self.round_started_event.invoke(self._name)

#     # 1) Trigger update events
#     def finish_round(self):

#         # self._module_progress.update_round(n_iterations)
#         # self._module_manager.update_progress(self._module_id, self._module_progress)
#         finished = self._teacher_progress.finish_round()
#         if finished: self._teacher_manager.update_progress(self._id, self._teacher_progress)
#         self.round_finished_event.invoke(self._name)

#     @property
#     def progress(self):

#         return self._teacher_progress.clone()


# class ExperimentManager(object):

#     STUDY_COL = 'ExperimentID'
#     PARAMS_COL = 'Params'
#     NAME_COL = 'Name'
#     AGGREGATE_FIELD_COL = 'Aggregates'
#     EVALUATION_FIELD_COL = 'Evaluations'
#     EXTENSION_COL = 'Extension'
#     FIELD_COL = 'Fields'
#     ID_COL = 'ID'

#     @dataclass
#     class ResultColumns(object):

#         columns: typing.List[str]

#     BASE_COLUMNS = [ID_COL, STUDY_COL, NAME_COL, PARAMS_COL, FIELD_COL, EXTENSION_COL]

#     def __init__(self):

#         self._df = pd.DataFrame(columns=self.BASE_COLUMNS)
#         # need to update this not to be private
#         self._module_manager = ModuleManager()
#         self._module_infos = {}
    
#     @property
#     def module_manager(self):
#         return self._module_manager

#     def create(self, name, study_name, params: None, result_fields, extension: ExtensionParams=None):
#         id = len(self._df)
#         result_columns = self.ResultColumns(result_fields)
#         extension = ExtensionParams()
#         self._df.loc[id, self.BASE_COLUMNS] = [id, name, study_name, params, result_columns, extension]
#         return id

#     def get(self, id):

#         return self._df.loc[id]


# class ExperimentInfo(object):

#     def __init__(
#         self, study_id: int, experiment_id: int, 
#         experiment_manager: ExperimentManager, 
#         params: ExtensionParams = None
#     ):
#         self._study_id: int = study_id
#         self._experiment_id: int = experiment_id
#         self._experiment_manager: ExperimentManager = experiment_manager
#         self._module_infos = {}
#         self._name_map = {}
#         self.params = params
    
#     # @property
#     # def learner(self):
#     #     return self._learner
    
#     # @property
#     # def fields(self):
#     #     return self._learner.aggregate_fields
    
#     # @property
#     # def evaluation_columns(self):
#     #     return self._learner.evaluation_fields
    
#     def create_teacher(self, name, teacher_params, n_rounds: int=1) -> TeacherInfo:
    
#         assert name not in self._name_map
#         progress = ModuleProgress(n_rounds=n_rounds)
#         id = self._experiment_manager.module_manager.create(
#             self._study_name, self._experiment_id, name, teacher_params, progress
#         )
#         self._teacher_infos[id] = TeacherInfo(
#             self._experiment_manager.module_manager, id, name, teacher_params, progress
#         )
#         self._name_map[name] = id
#         return self._teacher_infos[id]

#     # 1) Trigger update events
#     def get_teacher_info(self, id) -> TeacherInfo:
#         assert id in self._teacher_infos
#         return self._teacher_infos[id]
    
#     def get_teacher_info_by_name(self, name) -> TeacherInfo:
#         assert name in self._name_map
#         return self._module_accessors[self._name_map[name]]
    
#     # def update_learner(self, learner):
#     #     self._learner = learner


# class DojoManager(object):

#     pass


# class DojoInfo(object):

#     def create_experiment(self, name, result_fields: typing.List[str]):
#         pass
    
#     def get_experiment_info(self, id):
#         pass

#     def get_experiment_info_by_name(self, name):
#         pass


# class DataController(object):

#     def __init__(self):

#         self._dojo_manager = DojoManager()
#         self._dojo_infos = {}

#         self._default_dojo_manager = ExperimentManager()
#         self._default_experiment_infos = {}
#         self._name_map = {}
    
#     def create_dojo(self, name: str):

#         pass

#     def get_dojo_info(self, id: int):

#         pass

#     def get_dojo_info_by_name(self, name: str):

#         pass
    
#     def create_experiment(self, name, dojo_name: str, result_fields: typing.List[str], params: ExtensionParams) -> ExperimentAccessor:

#         assert name not in self._name_map
#         id = self._experiment_manager.create(
#             name, dojo_name, params, result_fields
#         )
#         self._experiment_accessors[id] = ExperimentInfo(
#             dojo_name, id, self._experiment_manager, params
#         )
#         self._name_map[name] = id
#         return self._experiment_accessors[id]

#     # 1) Trigger update events
#     def get_experiment_info(self, id) -> ExperimentInfo:
#         return self._experiment_accessors[id]
    
#     def get_experiment_info_by_name(self, name) -> ExperimentInfo:
#         assert name in self._name_map
#         return self._experiment_accessors[self._name_map[name]]
