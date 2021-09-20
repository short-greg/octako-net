from abc import abstractmethod
from dataclasses import dataclass, field as dataclass_field
from .bulletins import Class, Course, Goal, StandardGoal
from torch.utils.data.dataset import Dataset
from . import staff
from .staff import Run, Teacher, StandardTeacher
from . import observers
import typing
from torch.utils import data as data_utils



@dataclass
class Experiment(object):
    """[]
    """
    name: str
    goal: Goal
    n_steps: int
    base: staff.Ordered
    sub: staff.Staff
    audience: typing.Dict[str, Teacher]
    steps: int = 0
    is_finished: bool=False
    runs: typing.List[typing.Dict[str, Run]]=dataclass_field(default_factory=list)

    def take_step(self):
        self.steps += 1
    
    def finish(self):
        self.is_finished = True

    def evaluate(self) -> float:
        return self.goal.evaluate(self.runs)


# TODO: REFACTOR DOJO
# dojo should specify course etc


class Dojo(object):

    def __init__(
        self, name: str, goal: Goal=None
        
    ):
        """[A teaching system for training a learner]
        Args:
            name (str): [Name of the dojo]
            goal (Goal, optional): [The goal for the teaching system]. Defaults to None.
        """
        super().__init__()
        self._name: str = name
        self._goal: Goal = goal
        self._experiments: typing.List[Experiment] = []
        self._base: staff.Ordered = staff.Ordered()
        self._sub: staff.Staff = staff.Staff()
        self._audience: typing.Dict[str, observers.Observer] = {}
        self._n_runs: int = 0
        self._n_finished: int = 0

        self.course = Course(self._goal, 100)

    @property
    def goal(self):
        return self._goal
    
    @goal.setter
    def goal(self, goal: Goal):
        self._goal = goal

    def get(self, member: str, only_staff: bool=False):

        if member in self._base:
            return self._base[member]
        if member in self._sub:
            return self._sub[member]
        if not only_staff and member in self._audience:
            return self._audience[member]
        
        return None

    def get_base(self, idx: int):
        if idx < 0 or idx >= len(self._base):
            raise IndexError
        return self._base.get(idx)
    
    def is_staff(self, name: str) -> bool:
        """[Check if a name is a staff member]

        Args:
            name (str): [Name of staff member]

        Returns:
            bool: [Whether the name is a staff member]
        """

        return name in self._base or name in self._sub

    def is_observer(self, name: str) -> bool:
        """[Check if a name is an observer ]

        Args:
            member (str): [Name of the member]

        Returns:
            bool: [Whether the name is a staff member]
        """
        return name in self._audience

    def reorder(self, order: typing.List):
        """[Change the order of the base teachers]

        Args:
            order (typing.List): [The updated order for the base teachers]

        Returns:
            [type]: [description]
        """
        self._base.reorder(order)
        return self
    
    def check_addition(self, name):

        if name in self._audience or name in self._base or name in self._sub:
            raise KeyError(f'Member with name {name} already exists')
    
    def add_base_staff(self, teacher: Teacher):
        self.check_addition(teacher.name)
        teacher.bulletin_accessor = self.course.get_teacher_bulletin(teacher.name)
        self._base.add(teacher)
        return self
    
    def add_sub_staff(self, teacher: Teacher):
        self.check_addition(teacher.name)
        teacher.bulletin_accessor = self.course.get_teacher_bulletin(teacher.name)
        self._sub.add(teacher)
        return self
    
    def add_staff(self, teacher: Teacher, is_base: bool):

        if is_base:
            self.add_base_staff(teacher)
        else: self.add_sub_staff(teacher)
        return self

    def add_observer(self, observer: observers.Observer):
        self.check_addition(observer.name)
        self._audience[observer.name] = observer
        return self

    def summarize(self) -> str:

        # add __str__ function to all staff
        # summarize the base staff
        # sumamrize the sub staff
        # summarize the 

        return ""

    def __len__(self):
        return len(self._base)

    def experiment(self, learner, name: str='') -> Class:

        cur_class = self.course.start_class()
        student = cur_class.enroll(learner)
        self._n_runs += 1

        for i, teacher in enumerate(self._base):
            base_run: Run = teacher.run(student, f'{name}_{i}')
            # experiment.runs.append(base_run)
    
        # experiment.finish()
        self._n_finished += 1
        return cur_class


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
        trigger_builder: observers.TriggerBuilder
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
    name: str, to_maximize: bool, goal_field: str, training_data: Dataset, test_data: Dataset,
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
        observers.TriggerBuilder(
            "Validator Trigger"
        ).set_observing_round_finished().set_round_condition()
    ).get_result()


def build_testing_dojo(
    name: str, to_maximize: bool, goal_field: str, training_data: Dataset, test_data: Dataset,
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
        observers.TriggerBuilder(
            "Tester Trigger"
        ).set_finished_condition().set_observing_finished()
    ).build_progress_bar(
        "Progress Bar", listen_to=["Trainer", "Tester"]
    ).get_result()
