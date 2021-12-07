import dataclasses
import typing
from attr import field
from octako.machinery import learners
from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
from . import events
from torch.utils import data as torch_data
from . import observers
import typing
from typing import TypeVar, Generic


class IMessage(ABC):

    @abstractmethod
    def receive(self, recepient):
        pass

    @abstractmethod
    def get_response(self, recepient=None):
        pass


class IMessageReceiver(ABC):

    @abstractmethod
    def send(self, message: IMessage):
        pass


class EventSet:
    pass


class TeachingNode(IMessageReceiver):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def send(self, message: IMessage):
        pass

    @abstractproperty
    def events(self) -> EventSet:
        pass


class Teacher(TeachingNode):

    @property
    def name(self) -> str:
        pass

    @abstractmethod
    def teach(self):
        pass


class Observer(TeachingNode):
    
    @property
    def name(self) -> str:
        pass


@dataclasses.dataclass
class Evaluation:

    to_maximize: bool
    result: float


@dataclasses.dataclass
class Lecture:
    """[Struct for storing progress on a run]
    """
    name: str
    n_lessons: int
    n_lesson_iterations: int
    cur_lesson_iteration: int = 0
    cur_lesson: int = 0
    cur_iteration: int = 0
    finished: bool = False
    lesson_finished: bool = False
    results: pd.DataFrame = None

    def state_dict(self):
        # TODO: Implement
        pass

    def load_state_dict(self, state_dict):
        # TODO: Implement
        pass

    def __post_init__(self):
        self.results = self.results or pd.DataFrame()

    def append_results(self, results: typing.Dict[str, float]) -> bool:
        if self.finished:
            return False

        difference = set(results.keys()).difference(self.results.columns)
        for key in difference:
            self.results[key] = pd.Series(dtype='float')

        self.results.loc[len(self.results)] = results
        return True


class Course(ABC):

    def __init__(self):

        self.result_updated_event = events.TeachingEvent[str]("Result Updated")
        self.started_event = events.TeachingEvent[str]("Started")
        self.finished_event = events.TeachingEvent[str]("Finished")
        self.lesson_started_event = events.TeachingEvent[str]("Lesson Started")
        self.lesson_finished_event = events.TeachingEvent[str]("Lesson Finished")
        self.advance_event = events.TeachingEvent[str]("Advanced")
        self.message_posted_event = events.TeachingEvent[typing.Tuple[typing.Set[str], IMessage]]("Message Posted")

        self.EVENT_MAP = {
            "result_updated": self.result_updated_event,
            "started": self.started_event,
            "finished": self.finished_event,
            "lesson_started": self.lesson_started_event,
            "lesson_finished": self.lesson_finished_event,
            "advanced": self.advance_event,
            "message_posted": self.message_posted_event
        }

    def listen_to(self, event_key: str, f: typing.Callable[[str], typing.NoReturn], listen_to_filter: typing.Union[str, typing.Set[str]]=None):

        if event_key not in self.EVENT_MAP:
            raise ValueError(f'Event name {event_key} is not a valid event (Valid Events: {self.EVENT_MAP})')
        
        self.EVENT_MAP[event_key].add_listener(f, listen_to_filter)

    def state_dict(self):
        # TODO: Implement
        pass

    def load_state_dict(self, state_dict):
        # TODO: Implement
        pass

    @abstractmethod
    def update_results(self, teacher: Teacher):
        pass

    @abstractmethod
    def advance_lesson(self, teacher: Teacher):
        pass

    @abstractmethod
    def start_lesson(self, teacher: Teacher, n_lesson_iterations: int):
        pass

    @abstractmethod
    def start_lecture(self, teacher: Teacher, n_lessons: int):
        pass

    @abstractmethod
    def finish_lesson(self, teacher: Teacher):
        pass

    @abstractmethod
    def finish_lecture(self, teacher: Teacher):
        pass

    @abstractmethod
    def get_student(self, teacher: Teacher):
        pass

    @abstractmethod
    def get_cur_lecture(self, teacher_name: str) -> Lecture:
        pass
    
    @abstractmethod
    def evaluate(self) -> Evaluation:
        pass
    
    @abstractmethod
    def get_teacher(self, name: str) -> Teacher:
        pass

    @abstractmethod
    def get_observer(self, name: str) -> Observer:
        pass

    @abstractmethod
    def send_message(self, recepient: typing.Union[str, typing.List[str]], content: IMessage):
        """Send a message to another teacher or observer

        Args:
            recepient (typing.Union[str, typing.List[str]]): Name(s) of the recepients
            content (IMessage): The message sent
        """
        pass

    @abstractmethod
    def post_message(self, label: typing.Union[str, typing.Set[str]], content: IMessage):
        """Post a message for anyone to read

        Args:
            label (typing.Union[str, typing.Set[str]]): Label(s) for the message
            content (IMessage): The message posted
        """
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def resume(self):
        pass


class TeacherInviter(object):

    def __init__(self, teacher_cls: typing.Type[Teacher], teacher_name: str, *args, **kwargs):

        self._args = args
        self._kwargs = kwargs
        self._teacher_cls = teacher_cls
        self._teacher_name = teacher_name
    
    @property
    def teacher_name(self) -> str:
        return self._teacher_name

    def invite(self, course) -> Observer:
        
        return self._teacher_cls(*self._args, **self._kwargs, name=self._teacher_name, course=course)


class ObserverInviter(object):

    def __init__(self, observer_cls: typing.Type[Observer], observer_name: str, *args, **kwargs):

        self._args = args
        self._kwargs = kwargs
        self._observer_cls = observer_cls
        self._observer_name = observer_name
    
    @property
    def observer_name(self) -> str:
        return self._observer_name

    def invite(self, course) -> Observer:
        
        return self._observer_cls(*self._args, **self._kwargs, name=self._observer_name, course=course)


LearnerType = TypeVar('LearnerType', bound=learners.Learner)


class TeachingNetwork(ABC):
    
    @abstractmethod
    def teach(self, learner):
        pass
    
    @abstractmethod
    def resume_course(self, state_dict: dict):
        pass

    @abstractmethod
    def summarize(self):
        pass


class Goal(ABC):

    @abstractmethod
    def evaluate(self) -> Evaluation:
        pass
    
    @property
    def to_maximize(self) -> bool:
        pass


class GoalSetter(object):

    def __init__(self, goal_cls: typing.Type[Goal], *args, **kwargs):

        self._args = args
        self._kwargs = kwargs
        self._goal_cls = goal_cls

    def set(self, course) -> Goal:
        
        return self._goal_cls(*self._args, **self._kwargs, course=course)


class Staff(object):
    """[Convenience class for storing teachers]
    """

    def __init__(self):
        self._members = {}
    
    def add(self, teacher: Teacher):
        self._members[teacher.name] = teacher

    def get(self, name: str) -> typing.Union[Teacher, None]:
        return self._members.get(name)

    def __iter__(self):
        for teacher in self._members.values():
            yield teacher
    
    def __len__(self):
        return len(self._members)
    
    def __getitem__(self, name: str) -> Teacher:

        if name not in self._members:
            raise KeyError(f'{name} is not in staff.')

        return self._members[name]
    
    def __contains__(self, name: str):

        return name in self._members


class Ordered(Staff):
    """[Convenience class for storing staff an order. ]
    """

    def __init__(self):
        super().__init__()

        self._order = []
    
    def get(self, name: typing.Union[int, str]) -> Teacher:
        
        if type(name) == str:
            return super().get(name)
        return self._order[name]
    
    def add(self, teacher: Teacher):
        super().add(teacher)
        self._order.append(teacher)

    def reorder(self, order: typing.List):
        assert len(order) == len(self._order)
        assert len(set(order)) == len(order)
        assert max(order) == len(order) - 1
        assert min(order) == 0

        updated_order = []
        for i in order:
            updated_order.append(self._order[i])
        self._order = updated_order
    
    def __iter__(self):
        for teacher in self._order:
            yield teacher

    def __getitem__(self, name: typing.Union[str, int]):
        if type(name) == int:
            return self._order[name]
        return super().__getitem__(name)


class StandardCourse(Course):

    def __init__(self, learner: learners.Learner, goal_setter: GoalSetter):
        super().__init__()
        self._goal = goal_setter.set(self)
        self._lectures: typing.List[typing.Dict[str, typing.List[Lecture]]] = [{}]
        self._student = learner
        self._base_teachers: typing.Dict[str, Teacher] = dict()
        self._sub_teachers: typing.Dict[str, Teacher] = dict()
        self._observers: typing.Dict[str, Teacher] = dict()

    def get_cur_lecture(self, teacher_name: str) -> Lecture:
        if len(self._lectures) == 0 or teacher_name not in self._lectures[-1]:
            return None

        return self._lectures[-1][teacher_name][-1]
    
    def _verify_get_cur_lecture(self, teacher: Teacher) -> Lecture:

        lecture = self.get_cur_lecture(teacher.name)

        if lecture is None:
            raise ValueError("No current lecture for {}".format(teacher.name))
        return lecture

    def update_results(self, teacher: Teacher, results: typing.Dict[str, float]):  
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.append_results(results)
        self.result_updated_event.invoke(teacher.name, teacher.name)

    def start_lesson(self, teacher: Teacher, n_lesson_iterations: int):       
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.cur_lesson += 1
        lecture.cur_iteration = 0
        lecture.n_lesson_iterations = n_lesson_iterations
        self.lesson_started_event.invoke(teacher.name, teacher.name)

    def start_lecture(self, teacher: Teacher, n_lessons: int, n_lesson_iterations: int=0):
        current_lecture = self._lectures[-1]
        if teacher.name not in current_lecture:
            self._lectures[-1][teacher.name] = [Lecture(teacher.name, n_lessons, n_lesson_iterations)]
        else:
            self._lectures[-1][teacher.name].append(Lecture(teacher.name, n_lessons, n_lesson_iterations))

        self.started_event.invoke(teacher.name, teacher.name)

    def finish_lesson(self, teacher: Teacher):
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.lesson_finished = True
        self.lesson_finished_event.invoke(teacher.name, teacher.name)

    def finish_lecture(self, teacher: Teacher):
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.finished = True
        self.finished_event.invoke(teacher.name, teacher.name)

    def advance_lesson(self, teacher: Teacher):        
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.cur_iteration += 1
        lecture.cur_lesson_iteration += 1
        self.advance_event.invoke(teacher.name, teacher.name)

    def get_student(self, teacher: Teacher):
        return self._student

    def trigger_teacher(self, teacher_name):
        if teacher_name not in self._sub_teachers:
            raise ValueError("Teacher must be a sub teacher in order to trigger")
        self._sub_teachers[teacher_name].teach()
    
    def evaluate(self) -> Evaluation:
        return self._goal.evaluate()

    def set_teachers(self, base_teachers: typing.List[Teacher], sub_teachers: typing.List[Teacher], audience: typing.List[Observer]):
        self._base_teachers = {teacher.name: teacher for teacher in base_teachers}
        self._sub_teachers = {teacher.name: teacher for teacher in sub_teachers}
        self._audience = {observer.name: observer for observer in audience}
    
    def get_results(self, teacher_name: str, section_id: int=-1, lecture_id: int=-1):
        return self._lectures[section_id][teacher_name][lecture_id].results
    
    def get_base(self, name: str):
        if name in self._base_teachers:
            return self._base_teachers[name]

    def get_sub(self, name: str):
        if name in self._sub_teachers:
            return self._sub_teachers[name]

    def get_teacher(self, name) -> Teacher:
        return self.get_base(name) or self.get_sub(name) 

    def get_observer(self, name: str):
        if name in self._observers:
            return self._observers[name]

    def get_staff_member(self, name: str):
        return self.get_base(name) or self.get_sub(name) or self.get_observer(name)

    def send_message(self, recepients: typing.Union[str, typing.List[str]], message: IMessage):
        if isinstance(recepients, str):
            recepients = [recepients]
        
        for recepient_name in recepients:
            recepient = self.get_staff_member(recepient_name)
            if recepient is None:
                raise ValueError(f"Recepient {recepient_name} is not in the staff")
            recepient.send(message)            

    def post_message(self, labels: typing.Union[str, typing.Set[str]], message: IMessage):
        
        if isinstance(labels) == str:
            labels = set([labels])
        self.message_posted_event.invoke(labels, message)
    
    def state_dict(self):
        # TODO: Implement
        pass

    def load_state_dict(self, state_dict):
        # TODO: Implement
        pass

    # TODO: Consider if I really want start/resume in here or in the dojo
    def start(self):
        for i, (_, teacher) in enumerate(self._base_teachers.items()):
            teacher.teach()
            last_iteration = i == len(self._base_teachers) - 1
            if not last_iteration:
                self._lectures.append({})
    
    def resume(self):
        # TODO: Implement
        pass


class StandardGoal(Goal):

    def __init__(self, course: StandardCourse, to_maximize: bool, teacher_name: str, goal_field: str):

        self._course = course
        self._to_maximize = to_maximize
        self._teacher_name = teacher_name
        self._field = goal_field
    
    @property
    def to_maximize(self) -> bool:
        return self._to_maximize
    
    def evaluate(self) -> Evaluation:

        lecture_num = -1
        lesson_num = -1
        results: pd.DataFrame = self._course.get_results(self._teacher_name, lecture_num, lesson_num)
        return Evaluation(self._to_maximize, results[self._field].mean(axis=0))


class StandardTeacher(object):
    """Teacher that loops over the training data and calls the learn/test function on the learner
    """

    def __init__(
        self, name: str, course: Course, material: torch_data.Dataset, batch_size: int, n_lessons: int,
        is_training: bool=True,
    ):
        """[initializer]

        Args:
            name (str): [Name of the teacher]
            material (Dataset): [Default dataset for the teacher to use]
            batch_size (int): [Default batch size]
            n_rounds (int): [Default number of rounds to continue]
            is_training (bool, optional): [Whether the teacher should teach (learn) or test]. Defaults to True.
        """
        self._material = material
        self._is_training = is_training
        self._batch_size = batch_size
        self._name = name
        self._course = course
        self._n_lessons = n_lessons

    @property
    def name(self) -> str:
        return self._name
    
    def send(self, message: IMessage):
        # TODO: Implement
        pass

    def state_dict(self):
        # TODO: Implement
        pass

    def load_state_dict(self, state_dict):
        # TODO: Implement
        pass

    def teach(self):
        """[Run the teacher]

        Args:
            learner (learners.Learner): [Name of the learner]
            run_name (str): [Name of run]

        Returns:
            Run: The current run
        """
        to_shuffle = self._is_training

        data_loader = torch_data.DataLoader(
            self._material, batch_size=self._batch_size, 
            shuffle=to_shuffle
        )
        
        self._course.start_lecture(self, self._n_lessons)
        for _ in range(self._n_lessons):
            student = self._course.get_student(self)
            
            self._course.start_lesson(self, len(data_loader))

            for x, t in data_loader:
                if self._is_training:
                    item_results = student.learn(x, t)
                else:
                    item_results = student.test(x, t)
                item_results = {k: res.detach().cpu().numpy() for k, res in item_results.items()}
                self._course.update_results(self, item_results)
                self._course.advance_lesson(self)
            
            self._course.finish_lesson(self)
    
        self._course.finish_lecture(self)


class StandardTeacherInviter(TeacherInviter):


    def __init__(
        self, teacher_name: str, material: torch_data.Dataset, batch_size: int, n_lessons: int,
        is_training: bool=True
    ):
        super().__init__(
            StandardTeacher, teacher_name, material=material, batch_size=batch_size, 
            n_lessons=n_lessons, is_training=is_training
        )


class StandardTeachingNetwork(TeachingNetwork):

    def __init__(self, name: str, goal_setter: GoalSetter):
        self._name: str = name
        self._goal_setter: GoalSetter = goal_setter
        self._base: typing.List[TeacherInviter] = []
        self._sub: typing.Set[TeacherInviter] = set()
        self._members: typing.Set[str] = set()
        self._audience: typing.Set[ObserverInviter] = set()
        self._n_finished: int = 0

    def add_main_node(self, node: TeachingNode):
        pass

    def add_sub_node(self, node: TeachingNode, on_: typing.List[events.TeachingEvent]):
        pass
    
    def is_base_staff(self, name: str):
        for member in self._base:
            if name == member.teacher_name:
                return True

        return False

    def is_observer(self, name: str):
        for member in self._audience:
            if name == member.observer_name:
                return True

        return False

    def is_sub_staff(self, name: str):
        for member in self._sub:
            if name == member.teacher_name:
                return True

        return False
    
    def is_staff(self, name: str):
        return name in self._members

    def add_base_staff(self, teacher_inviter: TeacherInviter):

        if teacher_inviter.teacher_name in self._members:
            raise ValueError(f'Staff with name {teacher_inviter.teacher_name} already exists')
        
        self._members.add(teacher_inviter.teacher_name)
        self._base.append(teacher_inviter)

    def add_sub_staff(self, teacher_inviter: TeacherInviter):

        if teacher_inviter.teacher_name in self._members:
            raise ValueError(f'Staff with name {teacher_inviter.teacher_name} already exists')
        
        self._members.add(teacher_inviter.teacher_name)
        self._sub.add(teacher_inviter)

    def add_observer(self, observer_inviter: ObserverInviter):

        if observer_inviter.observer_name in self._members:
            raise ValueError(f'Staff with name {observer_inviter.observer_name} already exists')
        
        self._members.add(observer_inviter.observer_name)
        self._audience.add(observer_inviter)
    
    def get_teacher(self, name: str) -> typing.Tuple[TeacherInviter, bool]:

        for inviter in self._base:
            if name == inviter.teacher_name:
                return inviter, True
        
        for inviter in self._sub:
            if name == inviter.teacher_name:
                return inviter, False

        raise ValueError("Teacher name {} is not a valid teacher".format(name))
    
    def __len__(self):
        return len(self._base)
    
    def __getitem__(self, i):
        return self._base[i]
    
    def reorder(self, new_order: typing.List[int]):

        if len(new_order) != len(self._base):
            raise ValueError("New order must contain the same number of elements as the base")
        if min(new_order) != 0:
            raise ValueError("New order must have a lower bound of 0")
        if max(new_order) != len(self._base) - 1:
            raise ValueError("New order must have a upper bound equal to the length of the base teachers")
        if len(set(new_order)) != len(new_order):
            raise ValueError("New order must not contain duplicate values")
        self._base = [self._base[new_order[i]] for i in range(len(self._base))]
    
    def summarize(self) -> str:
        return ""
    
    def _load_course(self, learner: learners.Learner) -> StandardCourse:

        course = StandardCourse(learner, self._goal_setter)

        base_teachers = [teacher_inviter.invite(course) for teacher_inviter in self._base]
        sub_teachers = [teacher_inviter.invite(course) for teacher_inviter in self._sub]
        observers = [observer_inviter.invite(course) for observer_inviter in self._audience]

        course.set_teachers(
            base_teachers, sub_teachers, observers
        )
        return course
    
    # TODO: Remove "teach"
    def teach(self, learner: learners.Learner) -> StandardCourse:
        course = self._load_course(learner)
        course.start()
        
        return course
    
    # TODO: Remove "resume course"
    def resume_course(self, learner: learners.Learner, state_dict: dict):
        course = self._load_course(learner)
        course.load_state_dict(state_dict)
        course.resume()
        return course


class TeachingNetworkBuilder(object):
    """[Helper class for building a dojo. Simplfies some of the interactions]
    """

    def __init__(
        self, name: str, goal_setter: GoalSetter=None
    ):
        self._name = name
        self._goal_setter = goal_setter
        self.dojo = StandardTeachingNetwork(name, goal_setter)
    
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
        observers.TriggerInviter(
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
        observers.TriggerInviter(
            "Tester Trigger", "Trainer", "Tester"
        ).set_finished_condition().set_observing_finished()
    ).add_progress_bar(
        "Progress Bar", listen_to=["Trainer", "Tester"]
    ).get_result()


class Dojo(Generic[LearnerType], ABC):

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
