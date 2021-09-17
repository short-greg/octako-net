import pytest
from takonet.machinery import learners
from torch.utils import data as torch_data
import torch
from takonet.teaching import staff
import pandas as pd
import numpy as np


class DummyLearner(learners.Learner):

    def learn(self, x, t):
        return {'x': 1.0}
    
    def test(self, x, t):
        return {'x': 1.0}


def get_dataset():

    return torch_data.TensorDataset(
        torch.randn(2, 2)
    )


def get_teacher(name='Teacher'):

    return staff.StandardTeacher(
        name, get_dataset(), 32, 1, False
    )


class TestRun:

    def test_get_name(self):
        run = staff.Run(
            'run', DummyLearner(), get_dataset(), pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )
        assert run.name == 'run'

    def test_finish(self):
        run = staff.Run(
            'run', DummyLearner(), get_dataset(), pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )
        run.finish()

        assert run.progress.finished

    def test_finish_after_restarting(self):
        run = staff.Run(
            'run', DummyLearner(), get_dataset(), pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )
        run.finish()
        run.start()

        assert not run.progress.finished

    def test_finish_init_state(self):
        run = staff.Run(
            'run', DummyLearner(), get_dataset(), pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )

        assert not run.progress.finished

    def test_material(self):
        material = get_dataset()
        run = staff.Run(
            'run', DummyLearner(), material, pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )

        assert material is run.material

    def test_learner(self):
        learner = DummyLearner()
        run = staff.Run(
            'run', learner, get_dataset(), pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )

        assert learner is run.learner

    def test_update_results(self):
        run = staff.Run(
            'run', DummyLearner(), get_dataset(), pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )

        run.update(results={'x': 2.0})
        assert list(run.results.loc[0, 'x'])[0] == 2.0

    def test_update_results_with_two(self):
        run = staff.Run(
            'run', DummyLearner(), get_dataset(), pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )

        run.update(results={'x': 2.0})
        run.update(results={'x': 3.0})
        results = list(run.results.loc[1, 'x'])
        assert results[0] == 3.0

    def test_update_results_with_different_column(self):
        run = staff.Run(
            'run', DummyLearner(), get_dataset(), pd.DataFrame(columns=['x']), 32, "teacher",
            staff.RunProgress(10, 100)
        )

        run.update(results={'x': 2.0})
        run.update(results={'y': 3.0})
        results = run.results.loc[1, 'x']
        assert results is np.nan


class TestStaff:
    
    def test_add_teacher(self):

        base_staff = staff.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert base_staff.get(teacher.name) is teacher

    def test_get_nonexisting_teacher(self):

        base_staff = staff.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert base_staff.get("XXX") is None


    def test_index_with_existing(self):

        base_staff = staff.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert base_staff[teacher.name] is teacher

    def test_index_with_existing(self):

        base_staff = staff.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        with pytest.raises(KeyError):

            teacher = base_staff["XXX"]

    def test_index_with_nonexisting(self):

        base_staff = staff.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert base_staff[teacher.name] is teacher

    def test_teacher_in_staff(self):

        base_staff = staff.Staff()
        teacher = get_teacher()
        base_staff.add(teacher)

        assert teacher.name in base_staff


class TestOrdered:

    def test_get_after_adding_one(self):

        base_staff = staff.Ordered()
        teacher = get_teacher()
        base_staff.add(teacher)
        assert base_staff[0] == teacher

    def test_get_after_adding_two(self):

        base_staff = staff.Ordered()
        teacher = get_teacher()
        teacher2 = get_teacher('Teacher 2')
        base_staff.add(teacher2)
        base_staff.add(teacher)
        assert base_staff[0] == teacher2
        assert base_staff[1] == teacher

    def test_get_after_reordering(self):

        base_staff = staff.Ordered()
        teacher = get_teacher()
        teacher2 = get_teacher('Teacher 2')
        base_staff.add(teacher2)
        base_staff.add(teacher)
        base_staff.reorder([1, 0])
        assert base_staff[0] == teacher
        assert base_staff[1] == teacher2

    def test_getitem_with_str(self):

        base_staff = staff.Ordered()
        teacher = get_teacher()
        teacher2 = get_teacher('Teacher 2')
        base_staff.add(teacher2)
        base_staff.add(teacher)
        assert base_staff['Teacher 2'] is teacher2
        assert base_staff['Teacher'] is teacher

    def test_iter(self):

        base_staff = staff.Ordered()
        teacher = get_teacher()
        teacher2 = get_teacher('Teacher 2')
        base_staff.add(teacher2)
        base_staff.add(teacher)
        teachers = []
        for teach in base_staff:
            teachers.append(teach)
        assert teachers[0] is teacher2
        assert teachers[1] is teacher
