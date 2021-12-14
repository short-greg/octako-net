 
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, singledispatch, singledispatchmethod
from os import path
from typing import Any, Generic, TypeVar
import typing
import torch
from torch import nn
from .networks import Node, OpNode, Port


T = TypeVar('T')

class BaseVar(ABC):

    @property
    def value(self):
        raise NotADirectoryError


class Var(object):

    def __init__(self, name: str):
        
        self._name = name

    def process(self, kwargs):

        if self._name not in kwargs:
            raise ValueError("Value {self._name} not contained in kwargs")
        return kwargs

    def spawn(self, kwargs: dict):
        return Var(kwargs.get(self._name, self._name))


class Sz(object):

    def __init__(self, idx: int, port_idx: int=None):
        
        self._port_idx = port_idx
        self._idx = idx

    def process(self, size: torch.Size):

        if self._port_idx is None:
            return size[self._idx]
        return size[self._port_idx][self._idx]

@dataclass
class BuildPort(object):

    name: str
    idx: int


class OpFactory(ABC):

    @abstractmethod
    def produce(self, in_size: torch.Size, **kwargs) -> nn.Module:
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

    @abstractmethod
    def __getitem__(self, key) -> typing.List[BuildPort]:
        raise NotImplementedError


OpFactory.__call__ = OpFactory.spawn


# TODO: NEXT IMPLEMENT THIS

class Sequence(OpFactory):

    def __init__(self, op_factories: typing.List[OpFactory]):

        self._op_factories = op_factories
    
    def add(self, op_factory: OpFactory, position: int=None):

        pass


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
        return val.spawn(kwargs)

    @singledispatchmethod
    def _lookup_arg(self, val, in_size: torch.Size, kwargs):
        return val

    @_lookup_arg.register
    def _(self, val: Var, in_size: torch.Size, kwargs):
        return val.process(kwargs)

    @_lookup_arg.register
    def _(self, val: Sz, in_size: torch.Size, kwargs):
        return val.process(in_size)

    @abstractmethod
    def lookup(self, in_size: torch.Size, **kwargs):        
        raise NotImplementedError

    @abstractmethod
    def remap(self, **kwargs):    
        raise NotImplementedError


class Kwargs(_ArgMap):

    def __init__(self, **kwargs):

        self._kwargs = kwargs

    def lookup(self, in_size: torch.Size, **kwargs):
    
        return {k: self._lookup_arg(in_size, kwargs) for k, v in self._kwargs.items()}

    def remap(self, **kwargs):

        return {k: self._remap_arg(kwargs) for k, v in self._kwargs.items()}


class Args(_ArgMap):

    def __init__(self, *args):

        self._args = args

    def lookup(self, in_size: torch.Size, **kwargs):
    
        return [self._lookup_arg(in_size, kwargs) for v in self._args]

    def remap(self, **kwargs):

        return [self._remap_arg(kwargs) for v in self._args]


class BasicOp(OpFactory):

    def __init__(
        self, module: typing.Type[nn.Module], out: typing.Callable[[torch.Size], typing.List], args: Args, kwargs: Kwargs,
        labels: typing.List[str]=None, annotation: str=None
    ):

        self._module = module
        self._args = args
        self._kwargs = kwargs
        self._out = out
        self._labels = labels
        self._annotation = annotation
        
        # 1) determine where the variables are
        # 2) arg_vars, kwarg_vars (two varsets)
        # kwarg_vars[k]
    
    def __lshift__(self, other) -> Sequence:
        return Sequence([self, other])

    def produce(self, in_: typing.List[Port], **kwargs) -> nn.Module:
        my_kwargs = self._kwargs.lookup(in_, kwargs)
        my_args = self._args.lookup(in_, kwargs)
        return self._module(*my_args, **my_kwargs)
    
    def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
        
        my_kwargs = self._kwargs.lookup(in_, kwargs)
        my_args = self._args.lookup(in_, kwargs)
        module = self._module(*my_args, **my_kwargs)
        return OpNode(
            type(module).__name__, module, in_, self._out([i.size for i in in_])
        )

    def spawn(self, **kwargs):
        my_kwargs = self._kwargs.remap(kwargs)
        my_args = self._args.remap(kwargs)
        return BasicOp(
            self._module, self._out, my_args, my_kwargs
        )


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


class NullOut(Out):

    def spawn(self, **kwargs):
        return NullOut()

    def __call__(self, mod: nn.Module, in_size: torch.Size): 
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


class Mod(object):

    def __init__(self, mod: typing.Type[nn.Module], *args, **kwargs):

        self._module = mod
        self._args = Args(args)
        self._kwargs = Kwargs(kwargs)
    
    @singledispatchmethod
    def _out(self, out_size):
        if out_size is None:
            return NullOut()
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

    def produce(self, in_: typing.List[Port], **kwargs) -> nn.Module:
        my_kwargs = self._kwargs.lookup(in_, kwargs)
        my_args = self._args.lookup(in_, kwargs)
        return self._module(*my_args, **my_kwargs)


# mod(nn.Linear, 2, 3).op(out=(-1, Sz(1))
# mod(nn.Linear, 2, 3).op()
# linear = mod(nn.Linear, Var('x'), Var('y')).op(out=(-1, Sz(1))) << mod(nn.BatchNorm(Sz(1))) << mod(nn.Sigmoid)

# sequence = linear(x=2, y=3) << linear(x=3, y=4)
# 

# mod(nn.Conv2d, kw=2, kh=kl, stride=2, stride=3 ).op(fc)



# labels


class Mod(object):

    def __init__(self, module: typing.Type[nn.Module], *args, **kwargs):

        self._module = module
        self._args = args
        self._kwargs = kwargs

    def spawn(self, **kwargs):
        my_kwargs = self._kwargs.remap(kwargs)
        my_args = self._args.remap(kwargs)
        return BasicOp(
            self._module, self._out, my_args, my_kwargs
        )


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

