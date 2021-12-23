 
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from functools import partial, singledispatch, singledispatchmethod
from os import path
from typing import Any, TypeVar
import typing
import torch
from torch import nn
from torch import Size
from torch.nn import modules
from torch.nn.modules.container import Sequential
from .networks import Node, OpNode, Port
from octako.modules.containers import Parallel, Diverge


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


class PortSize(object):

    @singledispatchmethod
    def __init__(self, sizes: typing.List[torch.Size]):
        self._sizes = sizes
    
    @__init__.register
    def __init__(self, sizes: torch.Size):
        self._sizes = [sizes]


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
    
        return Kwargs(**{k: self._lookup_arg(v, in_size, kwargs) for k, v in self._kwargs.items()})

    def remap(self, kwargs):

        return Kwargs(**{k: self._remap_arg(v, kwargs) for k, v in self._kwargs.items()})

    @property
    def items(self):
        return self._kwargs


class Args(_ArgMap):

    def __init__(self, *args):

        self._args = args

    def lookup(self, in_size: torch.Size, kwargs):
    
        return Args(*[self._lookup_arg(v, in_size, kwargs) for v in self._args])

    def remap(self, kwargs):

        return Args(*[self._remap_arg(v, kwargs) for v in self._args])
    
    @property
    def items(self):

        return self._args


class ArgSet(_ArgMap):

    def __init__(self, *args, **kwargs):

        self._args = Args(*args)
        self._kwargs = Kwargs(*kwargs)
    
    def lookup(self, in_size: torch.Size, kwargs):

        return ArgSet(*self._args.lookup(in_size, kwargs).items, **self._kwargs.lookup(in_size, kwargs).items)
    
    def remap(self, kwargs):

        return ArgSet(*self._args.remap(kwargs).items, **self._kwargs.remap(kwargs).items)

    @property
    def kwargs(self):
        return self._kwargs.items

    @property
    def args(self):
        return self._args.items


class BaseMod(ABC):

    @abstractmethod
    def spawn(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def produce(self, in_: typing.List[Port], **kwargs):
        raise NotImplementedError

    @abstractproperty
    def module(self):
        raise NotImplementedError


class Out(ABC):

    @abstractmethod
    def spawn(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):
        raise NotImplementedError


# @dataclass
# class NodeMeta:
#     name: str=None
#     labels: typing.List[str] = field(default_factory=list)
#     annotation: str=None


class BasicOp(OpFactory):

    def __init__(
        self, module: BaseMod, 
        out: Out,
        name: str=None, labels: typing.List[str]=None, annotation: str=None
    ):
        if isinstance(out, torch.Size):
            self._out = SizeOut(out)
        else:
            self._out = out or NullOut()
        self._mod = module
        self._name = name
        self._labels = labels
        self._annotation = annotation
    
    def __lshift__(self, other) -> Sequence:
        return Sequence([self, other])

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Size):
            in_ = [in_]

        module = self._mod.produce(in_, **kwargs)
        return module, self._out.produce(module, in_, **kwargs)
    
    def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
        
        if isinstance(in_, Port):
            in_ = [in_]
        module = self._mod.produce([in_i.size for in_i in in_], **kwargs)
        name = self._name or type(module).__name__

        yield OpNode(
            name, module, in_, self._out.produce(module, in_)
        )
    
    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._name

    def spawn(self, **kwargs):
        mod = self._mod.spawn(**kwargs)
        return BasicOp(
            mod, self._out.spawn(**kwargs), self._name,
            self._labels, self._annotation
        )

    

ModType = typing.Union[typing.Type[nn.Module], Var]
ModInstance = typing.Union[nn.Module, Var]


class ModFactory(BaseMod):

    def __init__(self, module: ModType, args: ArgSet = None):

        self._module = module
        self._args = args or ArgSet

    def spawn(self, **kwargs):
        args = self._args.remap(kwargs)
        module = self._module.spawn(**kwargs) if isinstance(self._module, Var) else self._module
        return ModFactory(module, args)

    @singledispatchmethod
    def produce(self, in_: typing.List[torch.Size], **kwargs):
        
        module = self._module.spawn(**kwargs) if isinstance(self._module, Var) else self._module
        args = self._args.lookup(in_, kwargs)
        return module(*args.args, **args.kwargs)
    
    @produce.register
    def _(self, in_: torch.Size, **kwargs):
        return self.produce([in_], **kwargs)
    
    @property
    def module(self):
        return self._module

    @property
    def args(self):
        return self._args.args
    
    @property
    def kwargs(self):
        return self._args.kwargs

    def op(self, out: Out=None, name: str=None, labels: typing.List[str]=None, annotation: str=None) -> BasicOp:
        return BasicOp(self, out, name, labels, annotation)


class Instance(BaseMod):

    def __init__(self, module: ModInstance):

        self._module = module

    def spawn(self, **kwargs):
        module = self._module.spawn(**kwargs) if isinstance(self._module, Var) else self._module
        return module

    def produce(self, in_: typing.List[torch.Size], **kwargs):
        
        return self._module.spawn(**kwargs) if isinstance(self._module, Var) else self._module
    
    @property
    def module(self):
        return self._module

    def op(self, out: Out=None, name: str=None, labels: typing.List[str]=None, annotation: str=None) -> BasicOp:
        return BasicOp(self, out, name, labels, annotation)


def fc(mod: ModType, *args, **kwargs):
    return ModFactory(mod, ArgSet(*args, *kwargs))


def inst(mod: ModInstance):
    return Instance(mod)


class ListOut(Out):

    def __init__(self, size: typing.List[int]):
        
        self._size = Args(*size)

    def spawn(self, **kwargs):
        size = self._size.remap(kwargs)
        return ListOut(size)

    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs): 
        size = self._size.lookup(in_size, kwargs)
        return torch.Size(size.items)


class SizeOut(Out):

    def __init__(self, size: torch.Size):
        
        self._size =size
    
    def spawn(self, **kwargs):
        return SizeOut(self._size)

    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs): 
        return self._size


class NullOut(Out):

    def __init__(self):
        pass

    def spawn(self, **kwargs):
        return NullOut()

    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):         
        return in_size


class FuncOut(Out):

    def __init__(self, f: typing.Callable[[nn.Module, torch.Size, typing.Dict], torch.Size]):
        
        self._f = f

    def spawn(self, **kwargs):
        return FuncOut(self._f)

    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):         
        return FuncOut(mod, in_size, kwargs)


def _func_type():
    pass

_func_type = type(_func_type)


@singledispatch
def out(out_=None):
    if out_ is not None:
        raise ValueError(f'Argument out_ is not a valid type {type(out_)}')
    return NullOut()

@out.register
def _(out_: list):
    return ListOut(out_)


@out.register
def _(out_: _func_type):
    return FuncOut(out_)


@out.register
def _(out_: torch.Size):
    return SizeOut(out_)


class Override(OpFactory):

    pass


def over(factory: OpFactory):

    pass


class DivergeFactory(OpFactory):

    def __init__(
        self, op_factories: typing.List[OpFactory], 
        name: str=None, labels: typing.List[str]=None, annotation: str=None
    ):
        self._op_factories = op_factories
        self._name = name
        self._labels = labels
        self._annotation = annotation
    
    def __lshift__(self, other) -> Sequence:
        return Sequence([self, other])

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Size):
            in_ = [in_]
        mods = []
        outs = []
        for in_i, op_factory in zip(in_, self._op_factories):
            mod, out = op_factory.produce(in_i, **kwargs)
            mods.append(mod)
            outs.append(out)
        mod = Diverge(mods)
        return mod, outs
    
    def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
        
        if isinstance(in_, Port):
            in_ = [in_]
        
        for in_i, op_factory in zip(in_, self._op_factories):
            for node in op_factory.produce_nodes(in_i, **kwargs):
                yield node
    
    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._name

    def spawn(self, **kwargs):
        return DivergeFactory(
            [op_factory.spawn(**kwargs) for op_factory in self._op_factories],
            self._name, self._labels, self._annotation
        )


diverge = DivergeFactory


class ParallelFactory(OpFactory):

    def __init__(
        self, op_factories: typing.List[OpFactory], 
        name: str=None, labels: typing.List[str]=None, annotation: str=None
    ):
        self._op_factories = op_factories
        self._name = name
        self._labels = labels
        self._annotation = annotation
    
    def __lshift__(self, other) -> Sequence:
        return Sequence([self, other])

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Size):
            in_ = [in_]
        mods = []
        outs = []
        for op_factory in self._op_factories:
            mod, out = op_factory.produce(in_, **kwargs)
            mods.append(mod)
            outs.append(out)
        mod = Parallel(mods)
        return mod, outs
    
    def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
        
        if isinstance(in_, Port):
            in_ = [in_]
        
        for op_factory in self._op_factories:
            for node in op_factory.produce_nodes(in_, **kwargs):
                yield node
    
    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._name

    def spawn(self, **kwargs):
        return ParallelFactory(
            [op_factory.spawn(**kwargs) for op_factory in self._op_factories],
            self._name, self._labels, self._annotation
        )


parallel = ParallelFactory


class Chain(OpFactory):
    # need to allow for 
    def __init__(
        self, op_factory: OpFactory, attributes: typing.Union[Var, typing.List[Kwargs]],
        name: str=None, labels: typing.List[str]=None, annotation: str=None
    ):
        self._op_factory = op_factory
        self._attributes = attributes
        self._name = name
        self._labels = labels
        self._annotation = annotation
    
    def __lshift__(self, other) -> Sequence:
        return Sequence([self, other])

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Size):
            in_ = [in_]
        attributes = self._attributes
        if isinstance(attributes, Var):
            attributes = self._attributes.spawn(**kwargs)
        
        mods = []
        out = in_
        for attribute in self._attributes:
            mod, out = self._op_factory.produce(out, **attribute.items)
            mods.append(mod)
        return nn.Sequential(*mods), out
    
    def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
        
        if isinstance(in_, Size):
            in_ = [in_]
        attributes = self._attributes
        if isinstance(attributes, Var):
            attributes = self._attributes.spawn(**kwargs)
        
        for attribute in self._attributes:
            for node in self._op_factory.produce_nodes(in_, **attribute.items):
                yield node
            # Use produce to get the next in_size
            _, in_ = self._op_factory.produce(in_, **attribute.items)
    
    @property
    def labels(self):
        return self._labels

    @property
    def name(self):
        return self._name

    def spawn(self, **kwargs):
        attributes = self._attributes
        if isinstance(self._attributes, Var):
            attributes = self._attributes.spawn(**kwargs)
        return Chain(
            self._op_factory.spawn(kwargs), attributes,
            self._name, self._labels, self._annotation
        )


chain = Chain

class NetBuilder(object):
    pass


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


# class DefinedOp(OpFactory):

#     def __init__(
#         self, module: typing.Union[typing.Type[nn.Module], Var], out_size: typing.List[torch.Size], name: str=None, labels: typing.List[str]=None,
#         annotation: str=None
#     ):
#         self._var_module = VarModule(module)
#         self._out_size = out_size
#         self._annotation = annotation or ''
#         self._name = name
#         self._labels = labels or []
    
#     def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
#         var_module = self._var_module.produce(in_, **kwargs)
#         name = self._name or type(var_module.module).__name__
#         yield OpNode(name, var_module.module, in_, self._out_size, self._labels, self._annotation)

#     def produce(self, in_size: torch.Size, **kwargs) -> typing.Tuple[nn.Module, torch.Size]:
#         var_module = self._var_module.produce(in_size, **kwargs)
#         return var_module, self._out_size

#     def spawn(self, **kwargs):
#         var_module = self._var_module.spawn(**kwargs)

#         return DefinedOp(
#             var_module.module, self._out_size, self._name, self._labels, self._annotation
#         )




# class Mod(object):

#     def __init__(self, mod: typing.Type[nn.Module], *args, **kwargs):

#         self._module = mod
#         self._args = Args(*args)
#         self._kwargs = Kwargs(**kwargs)
    
#     @singledispatchmethod
#     def _out(self, out_size):
#         if out_size is None:
#             return null_out
#         if isinstance(out_size, _func_type):
#             return out_size
#         raise TypeError(f"Cannot process out_size of type {type(out_size).__name__}")
    
#     @_out.register
#     def _(self, out_size: tuple):
#         return TupleOut(out_size)

#     @_out.register
#     def _(self, out_size: Out):
#         return out_size

#     def op(self, out_size=None, name: str=None, labels: typing.List[str]=None, annotation: str=None) -> BasicOp:

#         # name = name or self._module.__name__
#         out_size = self._out(out_size)
#         return BasicOp(
#             self._module, out_size, self._args, self._kwargs, name=name, labels=labels, annotation=annotation
#         )

#     def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:
#         if isinstance(in_, Size):
#             in_ = [in_]
        
#         my_kwargs = self._kwargs.lookup(in_, kwargs)
#         my_args = self._args.lookup(in_, kwargs)
#         module = self._module.spawn(**kwargs) if isinstance(self._module, Var) else self._module
#         module = module(*my_args, **my_kwargs)
#         return module

# mod = Mod
