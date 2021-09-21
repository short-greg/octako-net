from takonet.teaching import dojo_builders, dojos
import torch
import torch.utils.data
from takonet.machinery import learners


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

        dojo_builder = dojo_builders.DojoBuilder("Validation Dojo", goal_setter)
        dojo_builder.add_trainer("Trainer", dataset, is_base=True)
        dojo = dojo_builder.get_result()
        assert dojo.is_base_staff("Trainer")

    def test_build_dojo_with_one_base_and_one_sub(self):

        goal_setter = dojos.GoalSetter(DummyGoal)

        dojo_builder = dojo_builders.DojoBuilder("Validation Dojo", goal_setter)
        dojo_builder.add_trainer("Trainer", dataset, is_base=True)
        dojo_builder.add_trainer("Trainer2", dataset, is_base=False)
        dojo = dojo_builder.get_result()
        assert dojo.is_base_staff("Trainer")
