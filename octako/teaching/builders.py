

from abc import ABC, abstractmethod
import dataclasses
import typing
from torch.utils import data as torch_data
from octako.machinery import learners
from octako.teaching.dojos import GoalSetter, LearnerType, ObserverInviter, StandardGoal, StandardTeacher, StandardTeachingNetwork, Teacher, TeacherInviter
from octako.teaching.observers import ProgressBar, TriggerInviter


class TeachingNetworkBuilder(object):
    """[Helper class for building a dojo. Simplfies some of the interactions]
    """

    def __init__(
        self, name: str, goal_setter: GoalSetter =None
    ):
        self._name = name
        self._goal_setter = goal_setter
        self.dojo = StandardTeachingNetwork (name, goal_setter)
    
    def add_staff(self, teacher: Teacher, is_base: bool=True):
        if is_base:
            self.dojo.add_base_staff(teacher)
        else:
            self.dojo.add_sub_staff(teacher)
        return self

    def add_tester(
        self, name: str, material: torch_data.Dataset, 
        batch_size: int=32, is_base: bool=False
    ):
        teacher = TeacherInviter(
            StandardTeacher, name, material=material, batch_size=batch_size, n_lessons=1
        )
        return self.add_staff(teacher, is_base)

    def add_trainer(
        self, name: str, material: torch_data.Dataset, n_lessons=10,
        batch_size: int=32, is_base: bool=False
    ):
        return self.add_staff(TeacherInviter(
            StandardTeacher, name, material=material, batch_size=batch_size, n_lessons=n_lessons,
        ), is_base)
    
    def add_teacher_trigger(self, name: str, trigger_inviter: TriggerInviter):

        self.dojo.add_observer(trigger_inviter)
        return self

    def add_progress_bar(self, name: str, listen_to: typing.List[str]):

        self.dojo.add_observer(
            ObserverInviter(ProgressBar, "Progress Bar", listen_to=listen_to)
        )
        return self
    
    def get_result(self):
        return self.dojo

    def reset(self, name: str=None, goal_setter: GoalSetter=None):

        name = name or self._name
        goal_setter = goal_setter or self._goal_setter

        self.dojo = StandardTeachingNetwork(name, goal_setter)
    

def build_validation_network(
    name: str, to_maximize: bool, goal_field: str, training_data: torch_data.Dataset, test_data: torch_data.Dataset,
    n_epochs: int=10, training_batch_size: int=32, test_batch_size: int=32
) -> StandardTeachingNetwork:
    """[Function to build a teaching network for doing validation]

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
    builder = TeachingNetworkBuilder(name, goal_setter)
    return builder.add_trainer(
        "Trainer", training_data, n_epochs, training_batch_size, True
    ).add_tester(
        "Validator", test_data, test_batch_size, False
    ).add_progress_bar(
        "Progress Bar", listen_to=["Trainer", "Validator"]
    ).add_teacher_trigger(
        "Validator Trigger",
        TriggerInviter(
            "Validator Trigger", "Trainer", "Validator"
        ).set_observing_lesson_finished().set_lesson_condition()
    ).get_result()


def build_testing_network(
    name: str, to_maximize: bool, goal_field: str, training_data: torch_data.Dataset, test_data: torch_data.Dataset,
    n_epochs: int=10, training_batch_size: int=32, test_batch_size: int=32
) -> StandardTeachingNetwork:    
    """[Function to build a teaching newtork for testing]

    Args:
        name (str): [Name of dojo]
        to_maximize (bool): [Whether the dojo should maximize]
        goal_field (str): [The goal for the testing]
        training_data (Dataset): [Dataset used for training]
        test_data (Dataset): [Dataset used for testing]
        n_epochs (int, optional): [Number of epochs to train for]. Defaults to 10.
        training_batch_size (int, optional): [Batch size for training]. Defaults to 32.
        test_batch_size (int, optional): [Batch Size for testing]. Defaults to 32.

    Returns:
        [Dojo]: The validation dojo
    """
    goal_setter = GoalSetter(StandardGoal, to_maximize=to_maximize, teacher_name="Tester", goal_field=goal_field)
    builder = TeachingNetworkBuilder(name, goal_setter)
    return builder.add_trainer(
        "Trainer", training_data, n_epochs, training_batch_size, True
    ).add_tester(
        "Tester", test_data, test_batch_size, False
    ).add_teacher_trigger(
        "Tester Trigger", 
        TriggerInviter(
            "Tester Trigger", "Trainer", "Tester"
        ).set_finished_condition().set_observing_finished()
    ).add_progress_bar(
        "Progress Bar", listen_to=["Trainer", "Tester"]
    ).get_result()


class Dojo(typing.Generic[LearnerType], ABC):

    @abstractmethod
    def validate(self, learner: LearnerType):
        raise NotImplementedError
    
    @abstractmethod
    def test(self, learner: LearnerType):
        raise NotImplementedError


@dataclasses.dataclass
class StandardDojo(Dojo[learners.Learner]):

    to_maximize: bool
    goal_field: str
    training_data: torch_data.Dataset
    validation_data: torch_data.Dataset
    final_training_data: torch_data.Dataset
    test_data: torch_data.Dataset

    name: str="Standard"
    n_validation_epochs: int=10
    n_testing_epochs: int=10
    training_batch_size: int=32
    test_batch_size: int=32

    def __post_init__(self):
        self._validation_network = build_validation_network(
            self.name, self.to_maximize, 
            self.goal_field, self.training_data, self.validation_data,
            self.n_validation_epochs, self.training_batch_size,
            self.test_batch_size 
        )

        self._testing_network = build_testing_network(
            self.name, self.to_maximize, 
            self.goal_field, self.final_training_data, self.test_data,
            self.n_testing_epochs, self.training_batch_size,
            self.test_batch_size 
        )

    # TODO: Decide on whether to keep these names
    def validate(self, learner: learners.Learner):
        
        return self._validation_network.teach(learner)

    def test(self, learner: learners.Learner):

        return self._testing_network.teach(learner)
