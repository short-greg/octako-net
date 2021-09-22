from .dojos import Dojo, GoalSetter, ObserverInviter, StandardDojo, StandardTeacher, Goal, StandardGoal, Teacher, TeacherInviter
from torch.utils import data as data_utils
from . import observers
import typing


class DojoBuilder(object):
    """[Helper class for building a dojo. Simplfies some of the interactions]
    """

    def __init__(
        self, name: str, goal_setter: GoalSetter=None
    ):
        self._name = name
        self._goal_setter = goal_setter
        self.dojo = StandardDojo(name, goal_setter)
    
    def add_staff(self, teacher: Teacher, is_base: bool=True):
        if is_base:
            self.dojo.add_base_staff(teacher)
        else:
            self.dojo.add_sub_staff(teacher)
        return self

    def add_tester(
        self, name: str, material: data_utils.Dataset, 
        batch_size: int=32, is_base: bool=False
    ):
        teacher = TeacherInviter(
            StandardTeacher, name, material=material, batch_size=batch_size, n_lessons=1
        )
        return self.add_staff(teacher, is_base)

    def add_trainer(
        self, name: str, material: data_utils.Dataset, n_lessons=10,
        batch_size: int=32, is_base: bool=False
    ):
        return self.add_staff(TeacherInviter(
            StandardTeacher, name, material=material, batch_size=batch_size, n_lessons=n_lessons,
        ), is_base)
    
    def add_teacher_trigger(self, name: str, trigger_inviter: observers.TriggerInviter):

        self.dojo.add_observer(trigger_inviter)
        return self

    def add_progress_bar(self, name: str, listen_to: typing.List[str]):

        self.dojo.add_observer(
            ObserverInviter(observers.ProgressBar, "Progress Bar", listen_to=listen_to)
        )
        return self
    
    def get_result(self):
        return self.dojo

    def reset(self, name: str=None, goal_setter: GoalSetter=None):

        name = name or self._name
        goal_setter = goal_setter or self._goal_setter

        self.dojo = Dojo(name, goal_setter)
    

def build_validation_dojo(
    name: str, to_maximize: bool, goal_field: str, training_data: data_utils.Dataset, test_data: data_utils.Dataset,
    n_epochs: int=10, training_batch_size: int=32, test_batch_size: int=32
) -> Dojo:
    """[Function to build a dojo for doing validation]

    Args:
        name (str): [Name of dojo]
        to_maximize (bool): [Whether the dojo should maximize]
        goal_field (str): [The goal for the validation]
        training_data (Dataset): [Dataset used for training]
        test_data (Dataset): [Dataset used for testing]
        n_epochs (int, optional): [Number of epochs to train for]. Defaults to 10.
        training_batch_size (int, optional): [Batch size for training]. Defaults to 32.
        test_batch_size (int, optional): [Batch Size for testing]. Defaults to 32.

    Returns:
        [Dojo]: The validation dojo
    """
    goal_setter = GoalSetter(StandardGoal, to_maximize=to_maximize, teacher_name="Validator", goal_field=goal_field)
    dojo_builder = DojoBuilder(name, goal_setter)
    return dojo_builder.add_trainer(
        "Trainer", training_data, n_epochs, training_batch_size, True
    ).add_tester(
        "Validator", test_data, test_batch_size, False
    ).add_progress_bar(
        "Progress Bar", listen_to=["Trainer", "Validator"]
    ).add_teacher_trigger(
        "Validator Trigger",
        observers.TriggerInviter(
            "Validator Trigger", "Trainer", "Validator"
        ).set_observing_lesson_finished().set_lesson_condition()
    ).get_result()


def build_testing_dojo(
    name: str, to_maximize: bool, goal_field: str, training_data: data_utils.Dataset, test_data: data_utils.Dataset,
    n_epochs: int=10, training_batch_size: int=32, test_batch_size: int=32
):    
    """[Function to build a dojo for doing validation]

    Args:
        name (str): [Name of dojo]
        to_maximize (bool): [Whether the dojo should maximize]
        goal_field (str): [The goal for the validation]
        training_data (Dataset): [Dataset used for training]
        test_data (Dataset): [Dataset used for testing]
        n_epochs (int, optional): [Number of epochs to train for]. Defaults to 10.
        training_batch_size (int, optional): [Batch size for training]. Defaults to 32.
        test_batch_size (int, optional): [Batch Size for testing]. Defaults to 32.

    Returns:
        [Dojo]: The validation dojo
    """
    goal_setter = GoalSetter(StandardGoal, to_maximize=to_maximize, teacher_name="Tester", goal_field=goal_field)
    dojo_builder = DojoBuilder(name, goal_setter)
    return dojo_builder.add_trainer(
        "Trainer", training_data, n_epochs, training_batch_size, True
    ).add_tester(
        "Tester", test_data, test_batch_size, False
    ).add_teacher_trigger(
        "Tester Trigger", 
        observers.TriggerInviter(
            "Tester Trigger", "Trainer", "Tester"
        ).set_finished_condition().set_observing_finished()
    ).add_progress_bar(
        "Progress Bar", listen_to=["Trainer", "Tester"]
    ).get_result()
