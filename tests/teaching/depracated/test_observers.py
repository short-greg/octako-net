from takonet.teaching import events
from takonet.machinery import learners
from takonet.teaching import observers
from takonet.teaching import staff
import pytest
import torch.utils.data as torch_data
import torch
import pandas as pd


class TestIterationCondition(object):

    def test_check_condition_with_valid(self):

        condition = observers.IterationCondition(period=2)
        progress = staff.RunProgress(10, 5, cur_iteration=1)
        assert condition.check(progress)

    def test_check_condition_with_invalid(self):

        condition = observers.IterationCondition(period=3)
        progress = staff.RunProgress(10, 5, cur_iteration=4)
        assert not condition.check(progress)

    def test_invalid_period(self):

        with pytest.raises(AssertionError):

            observers.IterationCondition(period=0)


class TestRoundCondition(object):

    def test_check_condition_with_valid(self):

        condition = observers.RoundCondition(period=2)
        progress = staff.RunProgress(10, 5, cur_round=3)
        assert condition.check(progress)

    def test_check_condition_with_invalid(self):

        condition = observers.RoundCondition(period=3)
        progress = staff.RunProgress(10, 5, cur_round=3)
        assert not condition.check(progress)

    def test_invalid_period(self):

        with pytest.raises(AssertionError):

            observers.RoundCondition(period=0)


class TestFinishedCondition(object):

    def test_check_condition_with_valid(self):

        condition = observers.FinishedCondition()
        progress = staff.RunProgress(10, 5, finished=True)
        assert condition.check(progress)

    def test_check_condition_with_valid(self):

        condition = observers.FinishedCondition()
        progress = staff.RunProgress(10, 5, finished=False)
        assert not condition.check(progress)


class Listener(object):

    def __init__(self):
        self.called = False
    
    def call(self):
        self.called = True


# class TestLearner(learners.Learner):

#     def learn(self, x, t):
#         return {}

#     def test(self, x, t): 
#         return {}
        

class TestTrigger(object):

    @staticmethod
    def setup_teacher() -> staff.StandardTeacher:

        return staff.StandardTeacher("teacher", torch_data.TensorDataset(torch.randn(4, 2)), 32, 1, 1, True)

    # def setup_run(self):

    #     return staff.Run(
    #         "x", TestLearner(), 
    #         torch_data.TensorDataset(torch.rand(2,2)), None, 32, "y", staff.RunProgress(2,2)
    #     )
        
    def test_on_trigger_not_executing_on_no_updates(self):
        listener = Listener()
        teaching_event = events.TeachingEvent[staff.Run]()
        progress = staff.RunProgress(10, 2, cur_iteration=0)
        run = staff.Run(
            'run', None, None, None, 0, 'teacher', progress
        )
        observers.Trigger('Teacher Trigger', observers.IterationCondition(2), teaching_event, listener.call)
        teaching_event.invoke(run)
        assert listener.called is False

    def test_on_trigger_executing_on_one_update(self):
        listener = Listener()
        teaching_event = events.TeachingEvent[staff.Run]()
        progress = staff.RunProgress(10, 2, cur_iteration=1)
        run = staff.Run(
            'run', None, None, None, 0, 'teacher', progress
        )
        observers.Trigger('Teacher Trigger', observers.IterationCondition(2), teaching_event, listener.call)
        teaching_event.invoke(run)
        assert listener.called is True


class TestProgressBar(object):

    def setup_data(self):
        return pd.DataFrame([[1, 2], [0, 1]], columns=['x', 'y'])

    def test_enter(self):

        teacher = staff.StandardTeacher("hi", None, None, 1, 1)
        pbar = observers.ProgressBar("PBar", [teacher])
        progress = staff.RunProgress(10, 2)
        run = staff.Run(
            'run', None, None, self.setup_data(), 0, 'teacher', progress
        )
        teacher.round_started_event.invoke(run)

        assert pbar.pbar is not None

    def test_update(self):

        teacher = staff.StandardTeacher("hi", None, None, 1, 1)
        pbar = observers.ProgressBar("PBar", [teacher])
        progress = staff.RunProgress(10, 1)
        run = staff.Run(
            'run', None, None, self.setup_data(), 0, 'teacher', progress
        )
        teacher.round_started_event.invoke(run)
        teacher.result_updated_event.invoke(run)
        assert pbar.pbar.n == 1

    def test_exit(self):

        teacher = staff.StandardTeacher("hi", None, None, 1, 1)
        pbar = observers.ProgressBar("PBar", [teacher])
        progress = staff.RunProgress(10, 1)
        run = staff.Run(
            'run', None, None, self.setup_data(), 0, 'teacher', progress
        )
        teacher.round_started_event.invoke(run)
        teacher.result_updated_event.invoke(run)
        teacher.round_finished_event.invoke(run)
        assert pbar.pbar is None