from abc import ABC, abstractmethod
from octako import learners
from . import dojos
from dataclasses import dataclass
import typing


@dataclass
class Study(ABC):

    @abstractmethod
    def perform(self) -> typing.List[dojos.Course]:
        pass


class LearnerBuilder(ABC):

    @abstractmethod
    def build(self) -> learners.Learner:
        pass


PDELIM = '/'
