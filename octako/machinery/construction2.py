 
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, singledispatch, singledispatchmethod
from os import path
from typing import Any, Callable, Generic, TypeVar
import typing
import torch
from torch import nn
from torch._C import Size
from torch.nn.modules.container import Sequential
from .networks import Node, OpNode, Port


T = TypeVar('T')

class BaseVar(ABC):

    @property
    def value(self):
        raise NotADirectoryError


class Var(object):

    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self):
        return self._name

    def spawn(self, **kwargs):
        return kwargs.get(self._name, self)


class Sz(object):

    def __init__(self, dim_idx: int, port_idx: int=None):
        
        self._port_idx = port_idx
        self._dim_idx= dim_idx

    def process(self, sizes: typing.List[torch.Size]):

        if self._port_idx is None:
            if len(sizes[0]) <= self._dim_idx:
                raise ValueError(f"Size dimension {len(sizes)} is smaller than the dimension index {self._dim_idx}.")
            return sizes[0][self._dim_idx]
        
        if len(sizes) <= self._port_idx:
            raise ValueError(f"Number of ports {len(sizes)} is smaller than the port index {self._port_idx}")

        if len(sizes[self._port_idx]) <= self._dim_idx:
            raise ValueError(f"Size dimension {len(sizes)} is smaller than the dimension index {self._dim_idx}")
        return sizes[self._port_idx][self._dim_idx]


@dataclass
class BuildPort(object):

    name: str
    idx: int


class OpFactory(ABC):

    @abstractmethod
    def produce(self, in_size: torch.Size, **kwargs) -> typing.Tuple[nn.Module, torch.Size]:
        raise NotImplementedError
    
    @abstractmethod
    def produce_nodes(self, in_size: torch.Size, **kwargs) -> typing.Iterator[Node]:
        raise NotImplementedError

    @abstractmethod
    def spawn(self, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def labels(self, labels):
        raise NotImplementedError

    @abstractmethod
    def name(self, name):
        raise NotImplementedError

    # @abstractmethod
    # def __getitem__(self, key) -> typing.List[BuildPort]:
    #     raise NotImplementedError


OpFactory.__call__ = OpFactory.spawn


class Sequence(OpFactory):

    def __init__(self, op_factories: typing.List[OpFactory]):

        self._op_factories = op_factories
    
    def add(self, op_factory: OpFactory, position: int=None):

        if position is None:
            self._op_factories.append(op_factory)
        else:
            self._op_factories.insert(op_factory, position)

    def __lshift__(self, other: OpFactory):
        return Sequence(self._op_factories + [other])
    
    def produce(self, in_: typing.List[torch.Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        sequential = nn.Sequential()

        for factory in self._op_factories:
            module, in_ = factory.produce(in_, **kwargs)
            sequential.add_module(type(module).__name__, module)
        return sequential, in_

    def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
        
        for factory in self._op_factories:
            for node in factory.produce_nodes(in_, **kwargs):      
                in_ = node.ports  
                yield node
    
    def spawn(self, **kwargs):
        return Sequence(
            [factory.spawn(**kwargs) for factory in self._op_factories]
        )

    @property
    def labels(self):
        return []

    @property
    def name(self):
        return ''


class NullFactory(OpFactory):

    # doesn't create a module
    pass


@abstractmethod
def _lshift(self, op_factory) -> Sequence:
    raise NotImplementedError

OpFactory.__lshift__ = _lshift


class _ArgMap(ABC):

    @singledispatchmethod
    def _remap_arg(self, val, kwargs):
        return val

    @_remap_arg.register
    def _(self, val: Var, kwargs):
        return val.spawn(**kwargs)

    @singledispatchmethod
    def _lookup_arg(self, val, in_size: torch.Size, kwargs):
        return val

    @_lookup_arg.register
    def _(self, val: Var, in_size: torch.Size, kwargs):
        return val.spawn(**kwargs)

    @_lookup_arg.register
    def _(self, val: Sz, in_size: torch.Size, kwargs):
        return val.process(in_size)

    @abstractmethod
    def lookup(self, in_size: torch.Size, kwargs):        
        raise NotImplementedError

    @abstractmethod
    def remap(self, kwargs):    
        raise NotImplementedError


class Kwargs(_ArgMap):

    def __init__(self, **kwargs):

        self._kwargs = kwargs

    def lookup(self, in_size: torch.Size, kwargs):
    
        return {k: self._lookup_arg(v, in_size, kwargs) for k, v in self._kwargs.items()}

    def remap(self, kwargs):

        return {k: self._remap_arg(v, kwargs) for k, v in self._kwargs.items()}


class Args(_ArgMap):

    def __init__(self, *args):

        self._args = args

    def lookup(self, in_size: torch.Size, kwargs):
    
        return [self._lookup_arg(v, in_size, kwargs) for v in self._args]

    def remap(self, kwargs):

        return [self._remap_arg(v, kwargs) for v in self._args]


def to_size(ports: typing.List[Port]):

    return [port.size for port in ports]


class BasicOp(OpFactory):

    def __init__(
        self, module: typing.Type[nn.Module], out: typing.Callable[[nn.Module, torch.Size], typing.List], args: Args=None, kwargs: Kwargs=None,
        name: str=None, labels: typing.List[str]=None, annotation: str=None
    ):
        if isinstance(out, torch.Size):
            self._out = lambda mod, sz: out
        else:
            self._out = out or null_out
        self._module = module
        self._args = args or Args()
        self._kwargs = kwargs or Kwargs()
        self._name = name or module.__name__
        self._labels = labels
        self._annotation = annotation
    
    def __lshift__(self, other) -> Sequence:
        return Sequence([self, other])

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Size):
            in_ = [in_]
        my_kwargs = self._kwargs.lookup(in_, kwargs)
        my_args = self._args.lookup(in_, kwargs)
        module = self._module(*my_args, **my_kwargs)
        return module, self._out(module, in_)
    
    def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
        
        if isinstance(in_, Port):
            in_ = [in_]
        in_ = to_size(in_)
        my_kwargs = self._kwargs.lookup(in_, kwargs)
        my_args = self._args.lookup(in_, kwargs)
        module = self._module(*my_args, **my_kwargs)
        
        yield OpNode(
            type(module).__name__, module, in_, self._out(module, in_)
        )
    
    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._name

    def spawn(self, **kwargs):
        my_kwargs = self._kwargs.remap(kwargs)
        my_args = self._args.remap(kwargs)
        return BasicOp(
            self._module, self._out, my_args, my_kwargs
        )

op = BasicOp


# @singledispatch
# def op(module_factory: typing.Callable[[], nn.Module], out: typing.List[torch.Size], name: str=None, labels: typing.List[str]=None, annotation: str=None):
#     pass


# @op.register
# def _(module, out: typing.List[torch.Size], *args, **kwargs):
#     return BasicOp(module, args, kwargs, out_size)


# @op.register
# def _(module, out: torch.Size, *args, **kwargs):
#     return BasicOp(module, args, kwargs, [out_size])


# @op.register
# def _(module, out: torch.Size, *args, **kwargs):
#     return BasicOp(module, args, kwargs, [out_size])

class Out(ABC):

    def spawn(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, mod: nn.Module, in_size: torch.Size):
        raise NotImplementedError



def null_out(mod: nn.Module, in_size: torch.Size): 
    return in_size


class TupleOut(Out):

    def __init__(self, size: typing.Tuple[int]):
        pass

    def spawn(self, **kwargs):
        my_kwargs = self._kwargs.remap(kwargs)
        my_args = self._args.remap(kwargs)
        return self.__class__

    def __call__(self, mod: nn.Module, in_size: torch.Size): 
        return in_size

def _func_type():
    pass

_func_type = type(_func_type)


class Mod(object):

    def __init__(self, mod: typing.Type[nn.Module], *args, **kwargs):

        self._module = mod
        self._args = Args(*args)
        self._kwargs = Kwargs(**kwargs)
    
    @singledispatchmethod
    def _out(self, out_size):
        if out_size is None:
            return null_out
        if isinstance(out_size, _func_type):
            return out_size
        raise TypeError(f"Cannot process out_size of type {type(out_size).__name__}")
    
    @_out.register
    def _(self, out_size: tuple):
        return TupleOut(out_size)

    @_out.register
    def _(self, out_size: Out):
        return out_size

    def op(self, out_size=None, name: str=None, labels: typing.List[str]=None, annotation: str=None) -> BasicOp:

        name = name or self._module.__name__
        out_size = self._out(out_size)
        return BasicOp(
            self._module, out_size, self._args, self._kwargs, labels, annotation
        )

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:
        if isinstance(in_, Size):
            in_ = [in_]
        
        my_kwargs = self._kwargs.lookup(in_, kwargs)
        my_args = self._args.lookup(in_, kwargs)
        module = self._module(*my_args, **my_kwargs)
        return module

mod = Mod

# mod(nn.Linear, 2, 3).op(out=(-1, Sz(1))
# mod(nn.Linear, 2, 3).op()
# linear = mod(nn.Linear, Var('x'), Var('y')).op(out=(-1, Sz(1))) << mod(nn.BatchNorm(Sz(1))) << mod(nn.Sigmoid)

# sequence = linear(x=2, y=3) << linear(x=3, y=4)
# 

# mod(nn.Conv2d, kw=2, kh=kl, stride=2, stride=3 ).op(fc)



# labels


# class Mod(object):

#     def __init__(self, module: typing.Type[nn.Module], *args, **kwargs):

#         self._module = module
#         self._args = args
#         self._kwargs = kwargs

#     def spawn(self, **kwargs):
#         my_kwargs = self._kwargs.remap(kwargs)
#         my_args = self._args.remap(kwargs)
#         return BasicOp(
#             self._module, self._out, my_args, my_kwargs
#         )


# override(linear, )


# class Override(OpFactory):

#     def __init__(self, op_factory: OpFactory):
        
#         self._op_factory = op_factory

#     def produce(self, in_: typing.List[Port], **kwargs) -> nn.Module:
#         pass

#     def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
#         pass

#     def remap(self, **kwargs):
#         pass


class Diverge(OpFactory):

    pass


class Override(OpFactory):

    pass


class Parallel(OpFactory):

    pass


# class Var(Generic[T]):

#     def __init__(self, value: T):
#         self._value = value

#     @property
#     def value(self) -> T:
#         return self._value


# class Shared(Generic[T]):

#     def __init__(self, var: Var[T]):
#         self._var = var

#     @property
#     def value(self) -> T:
#         return self._var.value

