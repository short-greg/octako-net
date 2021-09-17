from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
# from src.blueprints.utils import PDELIM
from src.teaching import data_controllers as study_controller
import optuna
import typing
from src.teaching import studies, params
from src.teaching import data_controllers as experiment_controller, learners as base_learner

# from src.materials import base as base_material


PDELIM = "/"


class ObjectiveRunner(object):

    def __init__(
        self, name, study_builder: studies.StudyBuilder, study_accessor: study_controller.StudyAccessor
    ):
        self.name = name
        self.study_builder = study_builder
        self.study_accessor = study_accessor
        self._cur_study = 0

    def __call__(self, trial):

        # experiment_accessor = self.study_accessor.create_experiment(f'{self.name} - Trial {self._cur_study}')

        # pass the name of the experiment
        learner, my_dojo, experiment_accessor = self.study_builder.build_for_trial(
            f'{self.name} - Trial {self._cur_study}', trial, self.study_accessor
        )
        my_dojo.run()

        # TODO: Evaluate results of experiment accessor
        self._cur_study += 1
        
        module_accessor = experiment_accessor.get_module_accessor_by_name(
            self.study_builder.optimize_module
        )

        results = module_accessor.get_results(
            retrieve_fields=[self.study_builder.optimize_field], 
            round_ids_filter=[module_accessor.progress.round]
        )
        return results.mean(axis=0).to_dict()[self.study_builder.optimize_field]


class FullStudy(studies.Study):

    def __init__(self, name: str, n_trials: int, study_builder: studies.StudyBuilder):
        self.study_builder = study_builder
        self.n_trials = n_trials
        self.name = name

    def run(self, name):
        # create the experiment accessor
        data_controller = study_controller.DataController()
        study_accessor = data_controller.create_study(self.name)

        objective = ObjectiveRunner(
            name, self.study_builder, study_accessor
        )

        my_study = optuna.create_study(direction=self.study_builder.direction)
        my_study.optimize(objective, n_trials=self.n_trials)

        # experiment_accessor = study_accessor.create_experiment("Best")
        
        learner, dojo, experiment_accessor = self.study_builder.build_best(
            "Best", params.ParamConverter(my_study.best_params).to_dict(), study_accessor
        )
        dojo.run()
