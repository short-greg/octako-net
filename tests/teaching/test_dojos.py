from octako.teaching import dojos
from octako import learners
import torch.utils.data
import torch
import pandas as pd
import numpy as np
import pytest


class LearnerTest(learners.Learner):

    fields = ['x']

    def learn(self, x, y):
        return {}

    def test(self, x, y):
        return {}


dataset = torch.utils.data.TensorDataset(
    torch.randn(4, 2), torch.randint(0, 4, (4,))
)


def get_teacher(name='Teacher'):

    return dojos.StandardTeacher(
        name, dataset, 32, 1, False
    )


class DummyLearner(learners.Learner):
    def learn(self, x, y): pass;
    def test(self, x, y): pass;


class DummyGoal(dojos.Goal):
    def evaluate(self): return 0.;


def create_lecture(n_lessons, cur_iteration):
    learner = DummyLearner()
    goal = dojos.Goal()
    return 


class TestLecture:

    def test_results_initializes_data_frame(self):
        lecture = dojos.Lecture("name", 1, 1)
        assert type(lecture.results) == pd.DataFrame

    def test_append_results_produces_correct_results(self):
        lecture = dojos.Lecture("name", 1, 1)
        result = {'y': 2, 'x': 3}
        lecture.append_results(result)
        assert lecture.results.loc[0, 'y'] == 2
        assert lecture.results.loc[0, 'x'] == 3

    def test_append_results_produces_correct_size(self):
        lecture = dojos.Lecture("name", 1, 1)
        result = {'y': 2, 'x': 3}
        lecture.append_results(result)
        assert len(lecture.results) == 1

    def test_append_results_produces_correct_result_with_different_columns(self):
        lecture = dojos.Lecture("name", 1, 1)
        result = {'y': 2, 'x': 3}
        result2 = {'y': 3, 'z': 9}
        lecture.append_results(result)
        lecture.append_results(result2)
        lecture.results.loc[0, 'y'] == 2
        lecture.results.loc[0, 'z'] == np.nan
        lecture.results.loc[0, 'x'] == 3
        lecture.results.loc[1, 'y'] == 3
        lecture.results.loc[1, 'z'] == 9
        lecture.results.loc[1, 'x'] == np.nan


class TestStaff:
    
    def test_add_teacher(self):

        base_staff = dojos.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert base_staff.get(teacher.name) is teacher

    def test_get_nonexisting_teacher(self):

        base_staff = dojos.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert base_staff.get("XXX") is None


    def test_index_with_existing(self):

        base_staff = dojos.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert base_staff[teacher.name] is teacher

    def test_index_with_existing(self):

        base_staff = dojos.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        with pytest.raises(KeyError):

            teacher = base_staff["XXX"]

    def test_index_with_nonexisting(self):

        base_staff = dojos.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert base_staff[teacher.name] is teacher

    def test_teacher_in_staff(self):

        base_staff = dojos.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert teacher.name in base_staff


class TestOrdered:

    def test_get_after_adding_one(self):

        base_staff = dojos.Ordered()
        teacher = get_teacher()
        base_staff.add(teacher)
        assert base_staff[0] == teacher

    def test_get_after_adding_two(self):

        base_staff = dojos.Ordered()
        teacher = get_teacher()
        teacher2 = get_teacher('Teacher 2')
        base_staff.add(teacher2)
        base_staff.add(teacher)
        assert base_staff[0] == teacher2
        assert base_staff[1] == teacher

    def test_get_after_reordering(self):

        base_staff = dojos.Ordered()
        teacher = get_teacher()
        teacher2 = get_teacher('Teacher 2')
        base_staff.add(teacher2)
        base_staff.add(teacher)
        base_staff.reorder([1, 0])
        assert base_staff[0] == teacher
        assert base_staff[1] == teacher2

    def test_getitem_with_str(self):

        base_staff = dojos.Ordered()
        teacher = get_teacher()
        teacher2 = get_teacher('Teacher 2')
        base_staff.add(teacher2)
        base_staff.add(teacher)
        assert base_staff['Teacher 2'] is teacher2
        assert base_staff['Teacher'] is teacher

    def test_iter(self):

        base_staff = dojos.Ordered()
        teacher = get_teacher()
        teacher2 = get_teacher('Teacher 2')
        base_staff.add(teacher2)
        base_staff.add(teacher)
        teachers = []
        for teach in base_staff:
            teachers.append(teach)
        assert teachers[0] is teacher2
        assert teachers[1] is teacher


class TestCourse:

    def test_get_cur_lecture_returns_none_if_none_for_teacher(self):

        course = dojos.StandardCourse(DummyLearner(), DummyGoal())
        assert course.get_cur_lecture("Teacher") is None

    def test_get_cur_lecture_returns_not_none_for_teacher_after_start(self):

        course = dojos.StandardCourse(DummyLearner(), DummyGoal())
        teacher = get_teacher()
        teacher2 = get_teacher("Teacher2")
        course.start(teacher, 5)
        course.advance(teacher)

        course.finish(teacher)
        course.start(teacher2, 5)
        
        assert course.get_cur_lecture(teacher.name) is not None

    def test_get_cur_lecture_returns_none_for_teacher_after_advance(self):

        course = dojos.StandardCourse(DummyLearner(), DummyGoal())
        teacher = get_teacher()
        teacher2 = get_teacher("Teacher2")
        course.start(teacher, 5)
        course.finish(teacher)
        course.advance_lecture()

        course.start(teacher2, 5)
        
        assert course.get_cur_lecture(teacher.name) is None

    def test_get_student(self):

        learner = DummyLearner()
        teacher = get_teacher()
        course = dojos.StandardCourse(learner, DummyGoal())
        assert course.get_student(teacher) == learner

    def test_evaluate(self):

        learner = DummyLearner()
        goal = DummyGoal()
        course = dojos.StandardCourse(learner, goal)
        assert course.evaluate() == goal.evaluate()

    def test_update_results_produces_correct_results(self):

        learner = DummyLearner()
        goal = DummyGoal()
        teacher = get_teacher()
        course = dojos.StandardCourse(learner, goal)
        course.start(teacher, 5, 2)
        course.update_results(teacher, {'x': 2})
        assert course.get_results(teacher.name).loc[0, 'x'] == 2


def get_teacher_inviter(name="Inviter", is_training=True):

    return dojos.StandardTeacherInviter(
        name, dataset, 32, 10, is_training
    )


class DummyGoalWithCourse(dojos.Goal):

    def __init__(self, course: dojos.Course):
        pass

    def evaluate(self): return 0.;


class DummyObserver(dojos.Observer):

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name


class TestDojo:

    def test_add_base_staff(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        dojo.add_base_staff(inviter)

        assert len(dojo) == 1
        assert dojo.is_base_staff(inviter.teacher_name)

    def test_add_sub_staff(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        dojo.add_sub_staff(inviter)

        assert len(dojo) == 0
        assert dojo.is_sub_staff(inviter.teacher_name)

    def test_is_staff_with_valid(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        dojo.add_base_staff(inviter)

        assert dojo.is_staff(inviter.teacher_name)

    def test_is_staff_with_non_staff_member(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        dojo.add_sub_staff(inviter)

        assert len(dojo) == 0
        assert not dojo.is_staff("Validator")

    def test_is_staff_with_sub(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        dojo.add_sub_staff(inviter)

        assert dojo.is_staff(inviter.teacher_name)

    def test_is_staff_with_observer(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        dojo.add_observer(dojos.ObserverInviter(DummyObserver, 'observer'))

        assert dojo.is_observer("observer")

    def test_is_not_observer_with_invalid(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        dojo.add_observer(dojos.ObserverInviter(DummyObserver, 'observer1'))

        assert not dojo.is_observer("observer")

    def test_reorder_cannot_be_done_if_duplicates(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        inviter2 = get_teacher_inviter(2)
        dojo.add_base_staff(inviter)
        dojo.add_base_staff(inviter2)

        with pytest.raises(ValueError):
            dojo.reorder([1, 1])

    def test_reorder_cannot_be_done_if_lower_bound_invalid(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        inviter2 = get_teacher_inviter(2)
        dojo.add_base_staff(inviter)
        dojo.add_base_staff(inviter2)

        with pytest.raises(ValueError):
            dojo.reorder([-1, 1])

    def test_reorder_cannot_be_done_if_upper_bound_invalid(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        inviter2 = get_teacher_inviter(2)
        dojo.add_base_staff(inviter)
        dojo.add_base_staff(inviter2)

        with pytest.raises(ValueError):
            dojo.reorder([0, 2])

    def test_reorder_produces_the_correct_order(self):

        dojo = dojos.StandardTeachingNetwork("dojo", dojos.GoalSetter(DummyGoalWithCourse))
        inviter = get_teacher_inviter()
        inviter2 = get_teacher_inviter(2)
        dojo.add_base_staff(inviter)
        dojo.add_base_staff(inviter2)

        dojo.reorder([1, 0])
        assert dojo[0].teacher_name == inviter2.teacher_name
        assert dojo[1].teacher_name == inviter.teacher_name


dataset = torch.utils.data.TensorDataset(
    torch.randn(4, 2), torch.randint(0, 4, (4,))
)


def get_teacher(name='Teacher'):

    return dojos.StandardTeacher(
        name, dataset, 32, 1, False
    )


class DummyLearner(learners.Learner):
    def learn(self, x, y): pass;
    def test(self, x, y): pass;


class DummyGoal(dojos.Goal):
    def evaluate(self): return 0.;


class TestDojoBuilder:

    def test_build_dojo_with_one_teacher(self):

        goal_setter = dojos.GoalSetter(DummyGoal)

        builder = dojos.TeachingNetworkBuilder("Validation Dojo", goal_setter)
        builder.add_trainer("Trainer", dataset, is_base=True)
        dojo = builder.get_result()
        assert dojo.is_base_staff("Trainer")

    def test_build_dojo_with_one_base_and_one_sub(self):

        goal_setter = dojos.GoalSetter(DummyGoal)

        builder = dojos.TeachingNetworkBuilder("Validation Dojo", goal_setter)
        builder.add_trainer("Trainer", dataset, is_base=True)
        builder.add_trainer("Trainer2", dataset, is_base=False)
        network = builder.get_result()
        assert network.is_base_staff("Trainer")

