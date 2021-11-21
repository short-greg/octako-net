from takonet.teaching import dojos
from . import staff
from takonet.machinery import learners
import torch.utils.data
import torch


class LearnerTest(learners.Learner):

    fields = ['x']

    def learn(self, x, y):
        return {}

    def test(self, x, y):
        return {}


dataset = torch.utils.data.TensorDataset(
    torch.randn(4, 2), torch.randint(0, 4, (4,))
)


class TestExperiment:

    def test_add_base_staff(self):

        dataset = torch.utils.data.TensorDataset(
            torch.randn(4, 2), torch.randint(0, 4, (4,))
        )

        builder = dojos.DojoBuilder(
            'Study', dojos.StandardGoal(True, "Teacher", 'X')
        )
        builder.build_trainer(
            'Teacher', dataset, 32, 5, True
        )

        # dojo.goal = dojos.Goal("Teacher", "Classification", True)
        assert len(builder.get_result()) == 1

    def test_add_sub_staff(self):

        dataset = torch.utils.data.TensorDataset(
            torch.randn(4, 2), torch.randint(0, 4, (4,))
        )

        builder = dojos.DojoBuilder(
            'Study', dojos.StandardGoal(True, "Teacher", 'X')
        )
        builder.build_trainer(
            'Teacher', dataset, 32, 10, True
        )
        builder.build_tester(
            'Validator', dataset, 32, False
        )
        # dojo.goal = dojos.Goal("Validator", "Classification", True)

        assert len(builder.get_result()) == 1

    def test_add_goal_with_valid(self):

        goal = dojos.StandardGoal("Teacher", "Classification", True)
        builder = dojos.DojoBuilder(
            'Study', goal
        )
        builder.build_trainer(
            'Teacher', dataset, 32, 5, True
        )
        # builder.goal = goal
        assert builder.get_result().goal is goal

    # def test_add_goal_with_invalid_module(self):
    # no longer performing any checks when setting the goal since i want
    # the goal to be free

    #     dojo = dojos.DojoBuilder(
    #         'Study', dojos.StandardGoal(True, "Teacher", 'X')
    #     )
    #     dojo.build_trainer(
    #         'Teacher', dataset, 32, 5, True
    #     )
    #     with pytest.raises(AssertionError):
    #         dojo.goal =dojos.StandardGoal("Validator", "Classification", True)
    
    def test_is_staff_with_valid(self):

        builder = dojos.DojoBuilder(
            'Study', dojos.StandardGoal(True, "Teacher", 'X')
        )
        builder.build_trainer(
            'Teacher', dataset, 32, 5, True
        )
        assert builder.get_result().is_staff('Teacher')

    def test_is_staff_with_non_staff_member(self):

        builder = dojos.DojoBuilder(
            'Study', dojos.StandardGoal(True, "Teacher", 'X')
        )
        builder.build_trainer(
            'Teacher', dataset, 32, 5, True
        )
        assert not builder.get_result().is_staff('Validator')

    def test_is_staff_with_sub(self):

        builder = dojos.DojoBuilder(
            'Study', dojos.StandardGoal(True, "Teacher", 'X')
        )
        builder.build_trainer(
            'Teacher', dataset, 32, 5, True
        )
        builder.build_tester(
            'Tester', dataset, 32, False
        )
        assert builder.get_result().is_staff('Tester')

    def test_is_staff_with_observer(self):

        dojo = dojos.DojoBuilder(
            'Study', dojos.StandardGoal(True, "Teacher", 'X')
        )
        dojo.build_trainer(
            'Teacher', dataset, 32, 5, True
        )

        dojo.build_progress_bar(
            'Progress Bar', ['Teacher']
        )
        assert not dojo.get_result().is_staff('Progress Bar')

    def test_is_observer_with_valid(self):

        builder = dojos.DojoBuilder(
            'Study', dojos.StandardGoal(True, "Teacher", 'X')
        )
        builder.build_trainer(
            'Teacher', dataset, 32, 5, True
        )

        builder.build_progress_bar(
            'Progress Bar', ['Teacher']
        )

        assert builder.get_result().is_observer('Progress Bar')

    def test_reorder(self):

        builder = dojos.DojoBuilder(
            'Study', dojos.StandardGoal(True, "Teacher", 'X')
        )
        builder.build_trainer(
            'Teacher', dataset, 32, 5, True
        )
        
        builder.build_trainer(
            'Teacher 2', dataset, 32, 5, True
        )
        dojo = builder.get_result()
        teacher = dojo.get_base(1)

        dojo.reorder([1, 0])

        assert dojo.get_base(0) is teacher
