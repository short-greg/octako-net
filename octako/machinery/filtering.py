from abc import ABC, abstractmethod
from functools import reduce
import typing
from octako.machinery.networks import Node, NodeSet


class Q(ABC):

    @abstractmethod
    def __call__(self, nodes: typing.Iterable[Node]) -> NodeSet:
        raise NotImplementedError
    
    def __or__(self, other):
        return OrQ([self, other])

    def __and__(self, other):
        return AndQ([self, other])

    def __sub__(self, other):
        return DifferenceQ([self, other])


class OrQ(Q):

    def __init__(self, queries: typing.List[Q]):

        self._queries = queries

    def __call__(self, nodes: typing.Iterable[Node]) -> NodeSet:
        
        node_sets = [query(nodes) for query in self._queries]
        return reduce(lambda x, y: x | y,  node_sets)

    def __or__(self, other):
        self._queries.append(other)
        return self


class AndQ(Q):

    def __init__(self, queries: typing.List[Q]):

        self._queries = queries

    def __call__(self, nodes: typing.Iterable[Node]) -> NodeSet:
        
        node_sets = [query(nodes) for query in self._queries]
        return reduce(lambda x, y: x & y,  node_sets)

    def __and__(self, other):
        self._queries.append(other)
        return self


class DifferenceQ(Q):

    def __init__(self, q1: Q, q2: Q):

        self._q1 = q1
        self._q2 = q2

    def __call__(self, nodes: typing.Iterable[Node]) -> NodeSet:
        
        return self._q1(nodes) - self._q2(nodes)
