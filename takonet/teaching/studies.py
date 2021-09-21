from abc import ABC, abstractmethod
from takonet.machinery import learners
from . import dojos
from dataclasses import dataclass
import typing


@dataclass
class Study(ABC):

    @abstractmethod
    def perform() -> typing.List[dojos.Course]:
        pass


class LearnerBuilder(ABC):

    @abstractmethod
    def build(self) -> dojos.Dojo:
        pass


PDELIM = '/'
