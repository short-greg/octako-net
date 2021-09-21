
import dataclasses
import typing

from torch._C import Value
from takonet.machinery import learners
from abc import ABC, abstractmethod
import pandas as pd
from . import events
from torch.utils import data as torch_data


class Teacher(ABC):

    @property
    def name(self) -> str:
        pass

    @abstractmethod
    def teach(self):
        pass


class Observer(ABC):
    
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


@dataclasses.dataclass
class Course(ABC):

    result_updated_event = events.TeachingEvent[str]()
    started_event = events.TeachingEvent[str]()
    finished_event = events.TeachingEvent[str]()
    lesson_started_event = events.TeachingEvent[str]()
    lesson_finished_event = events.TeachingEvent[str]()
    advance_event = events.TeachingEvent[str]()

    EVENT_MAP = {
        "result_updated": result_updated_event,
        "started": started_event,
        "finished": finished_event,
        "lesson_started": lesson_started_event,
        "lesson_finished": lesson_finished_event,
        "advanced": advance_event
    }

    def listen_to(self, event_key: str, f: typing.Callable[[str], typing.NoReturn]):

        if event_key not in self.EVENT_MAP:
            raise ValueError(f'Event name {event_key} is not a valid event (Valid Events: {self.EVENT_MAP})')
        
        self.EVENT_MAP[event_key].add_listener(f)

    @abstractmethod
    def update_results(self, teacher: Teacher):
        pass

    @abstractmethod
    def advance(self, teacher: Teacher):
        pass

    @abstractmethod
    def start_lesson(self, teacher: Teacher, n_lesson_iterations: int):
        pass

    @abstractmethod
    def start(self, teacher: Teacher, n_lessons: int):
        pass

    @abstractmethod
    def finish_lesson(self, teacher: Teacher):
        pass

    @abstractmethod
    def finish(self, teacher: Teacher):
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
    def advance_lecture(self):
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


class Dojo(ABC):
    
    @abstractmethod
    def teach(self, learner: learners.Learner):
        pass

    @abstractmethod
    def summarize(self):
        pass
        # create the course
        # invite teachers/observers


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

        self._goal = goal_setter.set(self)
        self._lectures: typing.List[typing.Dict[str, typing.List[Lecture]]] = [{}]
        self._student = learner
    
    def get_cur_lecture(self, teacher_name: str) -> Lecture:
        if len(self._lectures) == 0 or teacher_name not in self._lectures[-1]:
            return None
            # raise ValueError(f'Teacher {teacher_name} is not a part of the current lecture')

        return self._lectures[-1][teacher_name][-1]
    
    def _verify_get_cur_lecture(self, teacher: Teacher) -> Lecture:

        lecture = self.get_cur_lecture(teacher.name)

        if lecture is None:
            raise ValueError("No current lecture for {}".format(teacher.name))
        return lecture

    def update_results(self, teacher: Teacher, results: typing.Dict[str, float]):  
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.append_results(results)
        self.result_updated_event.invoke(teacher.name)

    def start_lesson(self, teacher: Teacher, n_lesson_iterations: int):       
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.cur_lesson += 1
        lecture.cur_iteration = 0
        lecture.n_lesson_iterations = n_lesson_iterations
        self.lesson_finished_event.invoke(teacher.name)

    def start(self, teacher: Teacher, n_lessons: int, n_lesson_iterations: int=0):
        if teacher.name not in self._lectures[-1]:
            self._lectures[-1][teacher.name] = [Lecture(teacher.name, n_lessons, n_lesson_iterations)]
        else:
            self._lectures[-1][teacher.name].append(Lecture(teacher.name, n_lessons, n_lesson_iterations))
        
        self.started_event.invoke(teacher.name)

    def finish_lesson(self, teacher: Teacher):
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.lesson_finished = True
        self.lesson_finished_event.invoke(teacher.name)

    def finish(self, teacher: Teacher):
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.finished = True
        self.finished_event.invoke(teacher.name)

    def advance(self, teacher: Teacher):        
        lecture = self._verify_get_cur_lecture(teacher)
        lecture.cur_iteration += 1
        lecture.cur_lesson_iteration += 1
        self.finished_event.invoke(teacher.name)

    def get_student(self, teacher: Teacher):
        return self._student
    
    def evaluate(self) -> Evaluation:
        return self._goal.evaluate()

    def advance_lecture(self):
        self._lectures.append({})
    
    def get_results(self, teacher_name: str, section_id: int=-1, lecture_id: int=-1):
        return self._lectures[section_id][teacher_name][lecture_id].results


class StandardGoal(Goal):

    def __init__(self, course: StandardCourse, is_maximization: bool, teacher_name: str, field: str):

        self._course = course
        self._is_maximization = is_maximization
        self._teacher_name = teacher_name
        self._field = field
    
    def evaluate(self) -> Evaluation:

        lecture_num = -1
        lesson_num = -1
        results: pd.DataFrame = self._course.get_results(self._teacher_name, lecture_num, lesson_num)
        return Evaluation(self._is_maximization, results[self._field].mean(axis=0))


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
        
        self._course.start(self, self._n_lessons)
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
                self._course.advance(self)
            
            self._course.finish_lesson(self)
    
        self._course.finish(self)


class StandardTeacherInviter(TeacherInviter):


    def __init__(
        self, teacher_name: str, material: torch_data.Dataset, batch_size: int, n_lessons: int,
        is_training: bool=True
    ):
        super().__init__(
            StandardTeacher, teacher_name, material=material, batch_size=batch_size, 
            n_lessons=n_lessons, is_training=is_training
        )


class StandardDojo(Dojo):

    def __init__(self, name: str, goal_setter: GoalSetter):
        self._name: str = name
        self._goal_setter: GoalSetter = goal_setter
        self._base: typing.List[TeacherInviter] = []
        self._sub: typing.Set[TeacherInviter] = set()
        self._members: typing.Set[str] = set()
        self._audience: typing.Set[ObserverInviter] = set()
        self._n_finished: int = 0
    
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
    
    def teach(self, learner: learners.Learner) -> StandardCourse:
        
        course = StandardCourse(learner, self._goal_setter)

        base_teachers = [teacher_inviter.invite(course) for teacher_inviter in self._base]
        sub_teachers = [teacher_inviter.invite(course) for teacher_inviter in self._sub]

        for observer_inviter in self._audience:
            observer_inviter.invite(course, sub_teachers)
        
        for teacher in base_teachers:
            teacher.teach()
        
        return course
