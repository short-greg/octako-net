from abc import ABC, abstractmethod
from copy import copy

from numpy import average
from takonet.machinery import learners
import typing
from . import students
import random
from dataclasses import dataclass
import pandas as pd


@dataclass
class RunProgress:
    """[Struct for storing progress on a run]
    """
    n_rounds: int = 0
    n_round_iterations: int = 0
    cur_round_iteration: int = 0
    cur_round: int = 0
    cur_iteration: int = 0
    finished: bool = False
    round_finished: bool = False

    def reset(self):

        self.cur_iteration = 0
        self.cur_round = 0
        self.cur_round_iteration = 0
        self.finished = False
        self.round_finished = False


@dataclass
class Entry(object):

    section: int
    id: int
    student: int
    teacher_name: str
    class_id: int
    progress: RunProgress
    results: pd.DataFrame = None

    def __post_init__(self):

        self.results = self.results or pd.DataFrame()
        self._finished = False

    def append_results(self, results: typing.Dict[str, float]) -> bool:
        if self.is_finished:
            return False

        difference = set(results.keys()).difference(self.results.columns)
        for key in difference:
            self.results[key] = pd.Series(dtype='float')

        self.results.loc[len(self.results)] = results
        return True

    def advance(self) -> bool:
        if self.is_finished:
            return False
        self.progress.cur_round_iteration += 1
        self.progress.cur_iteration += 1
        return True

    def start_round(self) -> bool:
        if self.is_finished:
            return False

        self.progress.cur_round += 1
        self.progress.cur_round_iteration = 0
        return True

    def finish_round(self) -> bool:
        if self.is_finished: return False
        self.progress.round_finished = True
        return True
    
    def reset(self) -> bool:
        if self.is_finished:
            return False
        
        self.progress.reset()
        self.results = pd.DataFrame()

        return True

    def finish(self) -> bool:
        if self.is_finished: return False
        self.progress.finished = True
        self._finished = True
        return True
        
    @property
    def is_finished(self):
        return self._finished



class Bulletin(object):

    def __init__(self):

        self._status_logs: typing.Dict[int, typing.List[typing.Dict[str, typing.List[Entry]]]] = {}
        self._cur_section_ids: typing.Dict[int, int] = dict()
    
    def _get_cur_section(self, class_id: int):
        cur_section_id = self._cur_section_ids[class_id]
        return self._status_logs[class_id][cur_section_id]
    
    def _get_section_key(self, teacher_name: str, student_id: int):
        return teacher_name, student_id

    def get_or_create_cur_entry(self, class_id: int, teacher_name: str, student_id: int) -> typing.Tuple[Entry, bool]:
        if class_id not in self._status_logs:
            self._status_logs[class_id] = [{}]
            self._cur_section_ids[class_id] = 0

        section_key = self._get_section_key(teacher_name, student_id)
        cur_section_id = self._cur_section_ids[class_id]

        cur_section = self._get_cur_section(class_id)
        if section_key not in cur_section:
            entry = Entry(cur_section_id, 0, student_id, teacher_name, class_id, RunProgress())
            cur_section[section_key] = [entry]
            return entry, True

        if self._get_cur_section(class_id)[section_key][-1].is_finished:
            run_id = len(cur_section[section_key])
            entry = Entry(cur_section_id, run_id, student_id, teacher_name, class_id, RunProgress())
            self._get_cur_section(class_id)[section_key].append(entry) 
            return entry, True
        
        return cur_section[section_key][-1], False

    def get_entry(self, class_id: int, teacher_name: str, student_id: int, section_id: int=-1, entry_id: int=-1) -> typing.Union[Entry, None]:

        section = self._status_logs[class_id][section_id]
        entries = section[self._get_section_key(teacher_name, student_id)]
        if entry_id is None:
            return entries
        return entries[entry_id]

    def get_n_sections(self, class_id: int):
        return len(self._status_logs[class_id])
    
    def get_entry_lists(self, class_id: int, section_id: int):
        return list(self._status_logs[class_id][section_id].keys())

    def get_n_entries(self, class_id: int, section_id: int, teacher_name: str, student_id: int):
        return len(self._status_logs[class_id][section_id][self._get_section_key(teacher_name, student_id)])

    def advance_section(self, class_id: int):
        if class_id not in self._cur_section_ids:
            self._cur_section_ids[class_id] = 0
        else:
            self._cur_section_ids[class_id] += 1
        if class_id not in self._status_logs:
            self._status_logs[class_id] = {}
    
        self._status_logs[class_id].append({})



class BulletinAccessor(object):

    def __init__(self, name: str, bulletin: Bulletin):

        self._name = name
        self._bulletin = bulletin

    def get_entry_accessor(self, class_id: int, student_id: int) -> Entry:

        entry, created = self._bulletin.get_or_create_cur_entry(class_id, self._name, student_id)
        return entry
    
    def get_entry(self, class_id: int, teacher_name: str, student_id: int, entry_id: typing.Union[int, None]=-1):

        return self._bulletin.get_entry(class_id, teacher_name, student_id, entry_id)


@dataclass
class Goal(ABC):
    """[Set a goal for an experiment]
    """
    to_maximize: bool

    @abstractmethod
    def evaluate(self, student_id: int, bulletin: Bulletin) -> float:
        pass

# TODO: Refactor how the class gets evaluated

@dataclass
class StandardGoal(Goal):
    """[Standard Goal]
    """
    teacher: str
    field: str
    class_id: int=None

    def evaluate(self, student_id: int, bulletin: Bulletin) -> float:
        return bulletin.get_entry(self.class_id, self.teacher, student_id).results[self.field].mean()


class Class(object):

    def __init__(self, class_id: int, bulletin: Bulletin, goal: Goal=None, max_students: int=100000):
        self._students: typing.Dict[int, students.Student] = {}
        self._class_id = class_id
        # TODO: Change this.. Shouldn't be a standard goal, but need to set the class
        self._goal: StandardGoal = copy(goal)
        if self._goal is not None:
            self._goal.class_id = class_id
        self._max_students = max_students
        self._bulletin = bulletin

    def has_exceeded_capacity(self):
        return len(self._students) == self._max_students

    def _choose_id(self) -> int:
        if self.has_exceeded_capacity():
            raise BufferError('Number of max students exceedeed')
        id = random.randint(0, self._max_students)
        if id in self._students:
            return self._choose_id()
        return id
    
    def get_student(self, id: int) -> students.Student:
        return self._students.get(id)

    def evaluate(self) -> typing.List[float]:
        if self._goal is None:
            return None
        return average([self._goal.evaluate(student.id, self._bulletin) for id, student in self._students.items()])

    def advance(self):
        self._bulletin.advance_section()
    
    def enroll(self, learner: learners.Learner) -> students.Student:

        id = self._choose_id()
        student = students.Student(learner, id, self._class_id)
        self._students[id] = student
        return student


class Course(object):

    def __init__(self, goal: Goal=None, max_students: int=1000000):
        self._classes: typing.List[Class] = []
        self._cur_id = 0
        self._max_students = max_students
        self._goal: Goal = goal
        self._bulletin: Bulletin = Bulletin()
    
    def start_class(self) -> Class:
        self._classes.append(Class(self._cur_id, self._bulletin, self._goal, self._max_students))
        return self._classes[-1]

    def get_class(self, class_id: int):

        return self._classes[class_id]

    def get_teacher_bulletin(self, name: str) -> BulletinAccessor:

        return BulletinAccessor(name, self._bulletin)
