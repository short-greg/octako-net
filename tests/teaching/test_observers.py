from takonet.teaching import events
from takonet.machinery import learners
from takonet.teaching import observers, dojos
import pytest
import torch.utils.data as torch_data
import torch
import pandas as pd


class DummyLearner(learners.Learner):
    def learn(self, x, y): pass;
    def test(self, x, y): pass;


class DummyGoal(dojos.Goal):
    def evaluate(self): return 0.;


def create_lecture(n_lessons, cur_iteration):
    learner = DummyLearner()
    goal = dojos.Goal()
    return 


class TestIterationCondition(object):

    def test_check_condition_with_valid(self):
        lecture = dojos.Lecture('lecture', n_lessons=5, n_lesson_iterations=5, cur_iteration=2)
        condition = observers.IterationCondition(period=2)
        assert condition.check(lecture)

    def test_check_condition_with_invalid(self):

        lecture = dojos.Lecture('lecture', n_lessons=10, n_lesson_iterations=5, cur_iteration=4)
        condition = observers.IterationCondition(period=3)
        assert not condition.check(lecture)

    def test_invalid_period(self):

        with pytest.raises(AssertionError):

            observers.IterationCondition(period=0)


class TestLessonCondition(object):

    def test_check_condition_with_valid(self):

        condition = observers.LessonCondition(period=2)
        progress = dojos.Lecture('lecture', 10, 5, cur_lesson=2)
        assert condition.check(progress)

    def test_check_condition_with_invalid(self):

        condition = observers.LessonCondition(period=2)
        progress = dojos.Lecture('lecture', 10, 5, cur_lesson=3)
        assert not condition.check(progress)

    def test_invalid_period(self):

        with pytest.raises(AssertionError):

            observers.LessonCondition(period=0)


class TestFinishedCondition(object):

    def test_check_condition_with_valid(self):

        condition = observers.FinishedCondition()
        lecture = dojos.Lecture('lecture', 10, 5, finished=True)
        assert condition.check(lecture)

    def test_check_condition_with_valid(self):

        condition = observers.FinishedCondition()
        lecture = dojos.Lecture('lecture', 10, 5, finished=False)
        assert not condition.check(lecture)


class Listener(object):

    def __init__(self):
        self.called = False
    
    def call(self):
        self.called = True


class StandardCourseForTesting(dojos.StandardCourse):

    def set_lecture(self, name: str, lecture: dojos.Lecture):
        if len(self._lectures) == 0:
            self._lectures.append({})
        self._lectures[-1][name] = [lecture]


def get_test_course():

    return StandardCourseForTesting(DummyLearner(), DummyGoal())


class TestTrigger(object):

    # @staticmethod
    # def setup_teacher() -> dojos.StandardTeacher:

    #    return dojos.StandardTeacher("teacher", torch_data.TensorDataset(torch.randn(4, 2)), 32, 1, 1, True)
        
    def test_on_trigger_not_executing_on_no_updates(self):
        listener = Listener()
        # teaching_event = events.TeachingEvent[str]()
        name = 'Teacher'
        course = get_test_course()
        course.set_lecture("Teacher", dojos.Lecture("Teacher", 5, 10, cur_iteration=1))
        
        observers.Trigger('Teacher Trigger', observers.IterationCondition(2), listener.call, course,  "advanced")
        
        course.advance_event.invoke(name)
        assert listener.called is False

    def test_on_trigger_executing_on_one_update(self):
        listener = Listener()
        # teaching_event = events.TeachingEvent[str]()
        name = 'Teacher'
        course = get_test_course()
        course.set_lecture("Teacher", dojos.Lecture("Teacher", 5, 10, cur_iteration=2))

        observers.Trigger('Teacher Trigger', observers.IterationCondition(2), listener.call, course,  "advanced")
        course.advance_event.invoke(name)
        assert listener.called is True
    