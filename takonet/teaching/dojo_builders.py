from .dojos import Dojo, StandardTeacher, Goal, StandardGoal, Teacher
from torch.utils import data as data_utils
from . import observers2 as observers
import typing


# move this out
class DojoBuilder(object):
    """[Helper class for building a dojo. Simplfies some of the interactions]
    """

    def __init__(
        self, name: str, goal: Goal=None
    ):
        self.dojo = Dojo(name, goal)

    def build_tester(
        self, name: str, material: data_utils.Dataset, 
        batch_size: int=32, is_base: bool=False
    ):
        self.dojo.add_staff(
            StandardTeacher(
                name, material, batch_size, 1
            ), is_base
        )
        return self

    def build_trainer(
        self, name: str, material: data_utils.Dataset, n_rounds=10,
        batch_size: int=32, is_base: bool=False
    ):
        self.dojo.add_staff(
            StandardTeacher(
                name, material, batch_size, n_rounds
            ), is_base
        )
        return self
    
    def build_teacher_trigger(
        self, listener: str, listening_to: str, 
        trigger_builder: observers.TriggerInviter
    ):
        listener: StandardTeacher = self.dojo.get(listener, only_staff=True)

        if listener is None:
            raise ValueError(f"{listener} is not a valid member of the dojo")
        listening_to: StandardTeacher = self.dojo.get(listening_to, only_staff=True)
        if listening_to is None:
            raise ValueError(f"{listening_to} is not a valid member of the dojo")
        assert listening_to is not None

        self.dojo.add_observer(trigger_builder.build(
                listening_to, listener.run, self.dojo.course
            ))
        return self

    def build_progress_bar(self, name: str, listen_to: typing.List[str]):

        teachers: typing.List[Teacher] = []
        for teacher_name in listen_to:
            teacher = self.dojo.get(teacher_name)
            assert teacher is not None
            teachers.append(teacher)

        self.dojo.add_observer(
            observers.ProgressBar(
                name, self.dojo.course, listen_to=teachers
            )
        )
        return self
    
    def get_result(self):
        return self.dojo

    def reset(self, name: str=None, goal: Goal=None):

        name = name or self.name
        goal = goal or self.goal

        self.dojo = Dojo(name, goal)
    

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
    goal = StandardGoal(to_maximize, "Validator", goal_field)
    dojo_builder = DojoBuilder(name, goal)
    return dojo_builder.build_trainer(
        "Trainer", training_data, n_epochs, training_batch_size, True
    ).build_tester(
        "Validator", test_data, test_batch_size, False
    ).build_progress_bar(
        "Progress Bar", listen_to=["Trainer", "Validator"]
    ).build_teacher_trigger(
        "Validator", "Trainer",
        observers.TriggerInviter(
            "Validator Trigger"
        ).set_observing_lesson_finished().set_round_condition()
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
    goal = StandardGoal(to_maximize, "Tester", goal_field)
    dojo_builder = DojoBuilder(name, goal)
    return dojo_builder.build_trainer(
        "Trainer", training_data, n_epochs, training_batch_size, True
    ).build_tester(
        "Tester", test_data, test_batch_size, False
    ).build_teacher_trigger(
        "Tester", "Trainer",
        observers.TriggerInviter(
            "Tester Trigger"
        ).set_finished_condition().set_observing_finished()
    ).build_progress_bar(
        "Progress Bar", listen_to=["Trainer", "Tester"]
    ).get_result()

