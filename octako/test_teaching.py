from sango.nodes import Status
import torch

from octako.learners import Learner, Validator
from . import teaching
import pytest
from torch.utils.data import TensorDataset



class TestResults:

    def test_results_with_no_data(self):

        results = teaching.Results()
        assert len(results.df) == 0

    def test_results_with_one_result(self):

        results = teaching.Results()
        progress = teaching.Progress("x", 1)
        results.add_result(
            "Validator", progress, {'x': 2}
        )
        assert len(results.df) == 1

    def test_results_with_two_results(self):

        results = teaching.Results()
        progress = teaching.Progress("x", 1)
        results.add_result(
            "Trainer", progress, {'x': 3, 'y': 3}
        )
        results.add_result(
            "Trainer", progress, {'x': 1}
        )
        assert len(results.df) == 2

    def test_results_with_two_result_columns(self):

        results = teaching.Results()
        progress = teaching.Progress("x", 1)
        results.add_result(
            "Trainer", progress, {'x': 3, 'y': 3}
        )
        results.add_result(
            "Trainer", progress, {'x': 1}
        )
        cols = results.result_cols
        assert 'x' in cols
        assert 'y' in cols

    def test_progress_cols_are_correct(self):

        results = teaching.Results()
        progress = teaching.Progress("x", 1)
        results.add_result(
            "Trainer", progress, {'x': 3, 'y': 3}
        )
        cols = results.progress_cols
        assert 'name' in cols
        assert 'total_epochs' in cols


class LearnerTest(Learner, Validator):

    def __init__(self):
        pass

    def learn(self, x, t):
        return {'loss': torch.tensor(1.0)}
    
    def test(self, x, t):
        return {'loss': torch.tensor(1.0)}


class TestTrain:

    def _create_teach(self):

        dataset = TensorDataset(
            torch.zeros(3, 2), torch.zeros(3)
        )

        return teaching.Train(
            name='str', 
            results=teaching.Results(), 
            dataset=dataset, batch_size=32, learner=LearnerTest(), 
            progress=teaching.ProgressRecorder(10)
        )

    def test_teach_init_works(self):
        teach = self._create_teach()
        assert isinstance(teach, teaching.Teach)

    def test_teach_tick_returns_running(self):
        teach = self._create_teach()
        assert teach.tick() == Status.RUNNING

    def test_teach_tick_returns_succcess(self):
        teach = self._create_teach()
        teach.tick()
        assert teach.tick() == Status.SUCCESS

    def test_teach_tick_returns_running_after_reset(self):
        teach = self._create_teach()
        teach.tick()
        teach.tick()
        teach.reset()
        assert teach.tick() == Status.RUNNING


class TestTest:

    def _create_teach(self):

        dataset = TensorDataset(
            torch.zeros(3, 2), torch.zeros(3)
        )

        return teaching.Validate(
            name='str', 
            results=teaching.Results(), 
            dataset=dataset, batch_size=32, learner=LearnerTest(), 
            progress=teaching.ProgressRecorder(10)
        )

    def _create_teach(self):

        dataset = TensorDataset(
            torch.zeros(3, 2), torch.zeros(3)
        )

        return teaching.Validate(
            name='str', 
            results=teaching.Results(), 
            dataset=dataset, batch_size=32, learner=LearnerTest(), 
            progress=teaching.ProgressRecorder(10)
        )

    def test_teach_init_works(self):
        teach = self._create_teach()
        assert isinstance(teach, teaching.Validate)

    def test_teach_tick_returns_running(self):
        teach = self._create_teach()
        assert teach.tick() == Status.RUNNING

    def test_teach_tick_returns_succcess(self):
        teach = self._create_teach()
        teach.tick()
        assert teach.tick() == Status.SUCCESS

    def test_teach_tick_returns_running_after_reset(self):
        teach = self._create_teach()
        teach.tick()
        teach.tick()
        teach.reset()
        assert teach.tick() == Status.RUNNING


class TestTrainer:

    def _create_teach(self):

        dataset = TensorDataset(
            torch.zeros(3, 2), torch.zeros(3)
        )

        return teaching.Trainer(
            name='Trainer',
            results=teaching.Results(), 
            validation_dataset=dataset, 
            training_dataset=dataset,
            learner=LearnerTest(),
            batch_size=32, 
            progress=teaching.ProgressRecorder(1)

        )

    def test_teach_init_works(self):
        teach = self._create_teach()
        assert isinstance(teach, teaching.Trainer)

    def test_teach_tick_returns_running(self):
        teach = self._create_teach()
        assert teach.tick() == Status.RUNNING

    def test_teach_tick_returns_succcess(self):
        teach = self._create_teach()
        teach.tick()
        teach.tick()
        teach.tick()
        teach.tick()
        teach.tick()
        teach.tick()
        teach.tick()
        assert teach.tick() == Status.SUCCESS

    # def test_teach_tick_returns_running_after_reset(self):
    #     teach = self._create_teach()
    #     teach.tick()
    #     teach.tick()
    #     teach.reset()
    #     assert teach.tick() == Status.RUNNING


