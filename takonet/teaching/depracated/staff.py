from abc import ABC, abstractmethod
from dataclasses import dataclass
import dataclasses
from .students import Student
from .bulletins import BulletinAccessor, Entry
import typing
from torch.utils.data.dataset import Dataset
import torch.utils.data as torch_data
from takonet.machinery import learners
import pandas as pd
from . import events


@dataclass
class RunProgress:
    """[Struct for storing progress on a run]
    """
    n_rounds: int
    n_round_iterations: int
    cur_round_iteration: int = 0
    cur_round: int = 0
    cur_iteration: int = 0
    finished: bool = False
    round_finished: bool = False


@dataclass
class Run(events.IResponse):
    """[Struct for storing the current run.]
    """
    learner: learners.Learner
    material: Dataset
    results: pd.DataFrame
    batch_size: int
    teacher_name: str
    progress: RunProgress
    is_training: bool=True
    shuffle: bool=True
    responses: typing.List[events.IResponse] = dataclasses.field(default_factory=list)

    def start_round(self):
        """[Start a new round]
        """
        self.progress.cur_round += 1
        self.progress.cur_round_iteration = 0
        self.progress.round_finished = False

    def finish_round(self):
        """[Finish the current round]
        """
        self.progress.round_finished = True

    def finish(self):
        """[Finish the run]
        """
        self.progress.finished = True

    def start(self):
        """[Start a new run]
        """
        self.progress.finished = False
    
    def update(self, results: typing.Dict[str, float]):
        """[Add results to the run]

        Args:
            results (typing.Dict[str, float]): [Results to add to the run]
        """
        self.progress.cur_iteration += 1
        self.progress.cur_round_iteration += 1
        self.results.loc[len(self.results), results.keys()] = results.values()
    
    def summarize(self) -> str:

        return ''
    
    def get_name(self) -> str:
        return self.teacher_name
    
    def add_response(self, response: events.IResponse):
        self.responses.append(response)


class Teacher(ABC):

    @property
    def results(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        pass


class StandardTeacher(object):
    """[summary]
    """

    ROUND_STARTED = 'round_started'
    ROUND_FINISHED = 'round_finished'
    RESULT_UPDATE = 'result update'
    STARTED = 'started'
    FINISHED = 'finished'
    EXTENSION_UPDATED = 'extension_updated'

    def __init__(
        self, name: str, material: Dataset, batch_size: int, n_rounds: int,
        is_training: bool=True, bulletin_accessor: BulletinAccessor=None
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
        self._n_round_iterations = len(material) // batch_size
        self._n_rounds = n_rounds
        
        self.bulletin_accessor = bulletin_accessor

        self.result_updated_event = events.TeachingEvent[Entry]()
        self.started_event = events.TeachingEvent[Entry]()
        self.started_round_event = events.TeachingEvent[Entry]()
        self.finished_event = events.TeachingEvent[Entry]()
        self.round_started_event = events.TeachingEvent[Entry]()
        self.round_finished_event = events.TeachingEvent[Entry]()

        self.run_event_map = {
            self.ROUND_STARTED: self.round_started_event,
            self.ROUND_FINISHED: self.round_finished_event,
            self.RESULT_UPDATE: self.result_updated_event,
            self.STARTED: self.started_event,
            self.FINISHED: self.finished_event,
        }

    @property
    def name(self) -> str:
        return self._name

    def run(self, student: Student, run_name: str) -> Run:
        """[Run the teacher]

        Args:
            learner (learners.Learner): [Name of the learner]
            run_name (str): [Name of run]

        Returns:
            Run: The current run
        """
        results = pd.DataFrame()

        if self.bulletin_accessor:
            entry = self.bulletin_accessor.get_entry_accessor(student.class_id, student.id)
        else:
            entry = Entry(0, 0, student.id, RunProgress(self._n_rounds, 0))

        run = Run(
            run_name, student, self._material, results , self._batch_size,
            self.name, RunProgress(self._n_rounds, n_round_iterations=1)
        )

        self.started_event.invoke(entry)
        data_loader = torch_data.DataLoader(
            run.material, batch_size=run.batch_size, 
            shuffle=run.shuffle
        )
        
        for _ in range(run.progress.n_rounds):
            entry.start_round()
            run.start_round()
            self.round_started_event.invoke(entry)
            entry.progress.n_round_iterations = len(data_loader)
            for x, t in data_loader:
                if run.is_training:
                    item_results = student.learn(x, t)
                else:
                    item_results = student.test(x, t)
                item_results = {k: res.detach().cpu().numpy() for k, res in item_results.items()}
                run.update(item_results)
                entry.append_results(item_results)
                entry.advance()
                self.result_updated_event.invoke(entry)
            
            run.finish_round()
            self.round_finished_event.invoke(entry)
    
        run.finish()
        entry.finish()
        self.finished_event.invoke(entry)
        return run


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
