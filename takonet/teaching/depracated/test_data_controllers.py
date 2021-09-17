import pytest
from . import data_controllers
from .data_controllers import ExtensionParams


class TestModuleProgress(object):

    def test_finish(self):

        progress = data_controllers.ModuleProgress()
        progress.finish()
        assert progress.finished is True

    def test_finished_after_start(self):

        progress = data_controllers.ModuleProgress()
        progress.finish()
        progress.start()
        assert progress.finished is False
        assert progress.turn == 1
    
    def test_start(self):

        progress = data_controllers.ModuleProgress()      
        assert progress.finished is False
        assert progress.turn == 0 
    
    def test_round_start(self):
        progress = data_controllers.ModuleProgress()
        progress.start()   
        result = progress.start_round(10)
        # round started is default state when created
        assert result is False and progress.round_finished is False
    
    def test_round_start_after_finish(self):
        progress = data_controllers.ModuleProgress()      
        progress.start_round(10) 
        progress.finish_round()     
        result = progress.start_round(20) 
        assert result is True and progress.round_finished is False and progress.n_iterations == 20

    def test_update_iteration(self):

        progress = data_controllers.ModuleProgress()
        progress.update_iteration()
        assert progress.iteration == 1
    
    def test_update_iteration_after_finish(self):
        progress = data_controllers.ModuleProgress()
        progress.start()
        progress.start_round()
        progress.update_iteration()
        progress.finish()
        result = progress.update_iteration()
        assert progress.iteration == 1 and result is False
        
    def test_update_iteration_after_round_finish(self):
        progress = data_controllers.ModuleProgress()
        progress.update_iteration()
        progress.finish_round()
        result = progress.update_iteration()
        assert progress.iteration == 1 and result is False

    def test_clone(self):
        progress = data_controllers.ModuleProgress()
        progress.update_iteration()
        progress.finish_round()
        progress.start_round()

        progress2 = progress.clone()
        result = progress2.update_iteration()
        assert progress2.iteration == 1 and progress2.round == 1
        assert progress.iteration == 0 and progress.round == 1


class TestExperimentAccessor(object):

    def build_experiment_accessor(self):

        return data_controllers.ExperimentAccessor(
            "Study", 0, 
            data_controllers.ExperimentManager(), ExtensionParams()
        )
    
    def test_create_module(self):

        experiment_accessor = self.build_experiment_accessor()
        module_accessor = experiment_accessor.create_module("Module1", ExtensionParams(), 10)
        assert type(module_accessor) == data_controllers.ModuleAccessor

    def test_create_existing_module(self):

        experiment_accessor = self.build_experiment_accessor()
        with pytest.raises(AssertionError):
            experiment_accessor.create_module("Module1", ExtensionParams())
            experiment_accessor.create_module("Module1", ExtensionParams())

    def test_get_module_by_name(self):

        experiment_accessor = self.build_experiment_accessor()
        experiment_accessor.create_module("Module1", ExtensionParams())
        module_accessor = experiment_accessor.get_module_accessor_by_name("Module1")
        assert type(module_accessor) == data_controllers.ModuleAccessor

    def test_get_nonexisting_module_by_name(self):

        experiment_accessor = self.build_experiment_accessor()
        experiment_accessor.create_module("Module1", ExtensionParams())
        with pytest.raises(AssertionError):
            experiment_accessor.get_module_accessor_by_name("Module2")


class Listener(object):

    def __init__(self):
        self.called = False

    def call(self, name):
        self.called = True


class TestModuleAccessor(object):

    def build_experiment_accessor(self):

        return data_controllers.ExperimentAccessor(
            "Study", 0, 
            data_controllers.ExperimentManager(), ExtensionParams()
        )
    
    def build_module(self):

        experiment_accessor = self.build_experiment_accessor()

        module_accessor = experiment_accessor.create_module("Module1", ExtensionParams(), n_rounds=2)
        return module_accessor

    def test_module_progress_is_initial(self):

        module_accessor = self.build_module()
        assert module_accessor.progress.turn == 0 and module_accessor.progress.round == 0

    def test_module_name(self):

        module_accessor = self.build_module()
        assert module_accessor.name == 'Module1'

    def test_start(self):

        listener = Listener()

        module_accessor = self.build_module()
        module_accessor.started_event.add_listener(listener.call)
        module_accessor.start()
        assert listener.called is True

    def test_start_round(self):

        listener = Listener()

        module_accessor = self.build_module()
        module_accessor.round_started_event.add_listener(listener.call)
        module_accessor.start()
        assert listener.called is False
        module_accessor.start_round()
        assert listener.called is True

    def test_finish_round(self):

        listener = Listener()

        module_accessor = self.build_module()
        module_accessor.round_finished_event.add_listener(listener.call)
        module_accessor.start_round()
        module_accessor.finish_round()
        assert listener.called is True

    def test_add_results_listener(self):

        listener = Listener()

        module_accessor = self.build_module()
        module_accessor.result_updated_event.add_listener(listener.call)
        module_accessor.add_result({'X': 2, 'Y': 10})
        assert listener.called is True

    def test_get_results_gets_correct_value(self):

        module_accessor = self.build_module()
        module_accessor.add_result({'X': 2, 'Y': 10})
        result = module_accessor.get_results(['X'])
        assert result.loc[0, 'X'] == 2

    def test_get_results_doesnt_retrieve_y(self):

        module_accessor = self.build_module()
        module_accessor.add_result({'X': 2, 'Y': 10})
        result = module_accessor.get_results(['X'])
        with pytest.raises(KeyError):
            result.loc[0, 'Y']

    def test_get_results_gets_correct_round(self):

        module_accessor = self.build_module()
        module_accessor.start_round()
        module_accessor.finish_round()
        module_accessor.start_round()
        module_accessor.add_result({'X': 2, 'Y': 10})
        result = module_accessor.get_results(['X'], round_ids_filter=[1])
        assert result.loc[0, 'X'] == 2

    def test_get_results_gets_correct_iterations(self):

        module_accessor = self.build_module()
        module_accessor.start_round()
        module_accessor.add_result({'X': 10, 'Y': 10})
        module_accessor.finish_round()
        module_accessor.start_round()
        module_accessor.add_result({'X': 2, 'Y': 10})
        module_accessor.add_result({'X': 3, 'Y': 9})
        module_accessor.add_result({'X': 4, 'Y': 11})
        result = module_accessor.get_results(['X'], round_ids_filter=[1], iteration_ids_filter=list(range(1,3)))
        assert result.iloc[0]['X'] == 3

from src.teaching.data_controllers import ExperimentAccessor, ExtensionParams
from . import data_controllers
import pytest


class TestDataController:

    # def test_create_study_with_new_study(self):

    #     controller = data_controllers.DataController()
    #     study_accessor = controller.create_study("Study1")
    #     assert type(study_accessor) == data_controllers.StudyAccessor

    # def test_create_study_with_existing_study(self):

    #     controller = data_controllers.DataController()
    #     with pytest.raises(AssertionError):
    #         controller.create_study("Study1")
    #         controller.create_study("Study1")

    def test_create_experiment_with_new_experiment(self):
        controller = data_controllers.DataController()
        experiment_accessor = controller.create_experiment("Experiment1", "Study", ['X'], ExtensionParams())
        assert type(experiment_accessor) == data_controllers.ExperimentAccessor

    def test_create_experiment_with_existing_experiment(self):
        controller = data_controllers.DataController()
        with pytest.raises(AssertionError):
            controller.create_experiment("Experiment1", "Study", ['X'], ExtensionParams())
            controller.create_experiment("Experiment1", "Study", ['X'], ExtensionParams())

    # def test_get_study_accessor_with_existing(self):
    #     controller = data_controllers.DataController()
    #     controller.create_study("Study1")
    #     study_accessor = controller.get_study_accessor_by_name("Study1")
    #     assert type(study_accessor) == data_controllers.StudyAccessor

    # def test_get_study_accessor_with_not_existing(self):
    #     controller = data_controllers.DataController()
    #     with pytest.raises(AssertionError):
    #         controller.get_study_accessor_by_name("Study1")

    def test_get_experiment_accessor_with_existing(self):
        controller = data_controllers.DataController()
        controller.create_experiment("Experiment1", "Study", ['X'], ExtensionParams())
        experiment_accessor = controller.get_experiment_accessor_by_name("Experiment1")
        assert type(experiment_accessor) == data_controllers.ExperimentAccessor

    def test_get_experiment_accessor_with_not_existing(self):
        controller = data_controllers.DataController()
        with pytest.raises(AssertionError):
            controller.get_experiment_accessor_by_name("Experiment1")


# class TestStudyAccessor:

#     def test_create_experiment_with_new(self):
#         controller = data_controllers.DataController()
#         study = controller.create_study("Study1")
#         experiment = study.create_experiment("Experiment1", None, None)
#         assert type(experiment) == ExperimentAccessor

#     def test_create_experiment_with_existing(self):
#         controller = data_controllers.DataController()
#         study = controller.create_study("Study1")
#         with pytest.raises(AssertionError):
#             study.create_experiment("Experiment1", None, None)
#             study.create_experiment("Experiment1", None, None)

#     def test_get_experiment_accessor_with_existing(self):
#         controller = data_controllers.DataController()
#         study_accessor = controller.create_study("Study1")
#         study_accessor.create_experiment("Experiment1", None, None)
#         experiment_accessor = study_accessor.get_experiment_accessor_by_name("Experiment1")
#         assert type(experiment_accessor) == ExperimentAccessor

#     def test_get_experiment_accessor_with_not_existing(self):
#         controller = data_controllers.DataController()
#         study_accessor = controller.create_study("Study1")
#         with pytest.raises(AssertionError):
#             study_accessor.get_experiment_accessor_by_name("Experiment1")
