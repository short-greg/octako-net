 
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from functools import partial, singledispatch, singledispatchmethod
from os import path
from typing import Any, Callable, Counter, TypeVar
import typing
import torch
from torch import nn
from torch import Size
from torch.nn import modules
from torch.nn.modules.container import Sequential
from .networks import In, ModRef, Multitap, Network, Node, NodeSet, OpNode, Parameter, Port
from octako.modules.containers import Parallel, Diverge


T = TypeVar('T')

class BaseVar(ABC):

    @property
    def value(self):
        raise NotADirectoryError


class var(object):

    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self):
        return self._name

    def to(self, **kwargs):
        return kwargs.get(self._name, self)


#         # class SizeMeta(type):

#         #     def __call__(cls, *args, **kwargs):

#         #         obj = cls.__new__(cls, *args, **kwargs)
#         #         obj.__init__(*args, **kwargs)
#         #         return obj

#         #     def __getitem__(cls, idx: int):
#         #         return cls(idx)
            
#         # class Size(object, metaclass=SizeMeta):

#         #     def __init__(self, idx):
#         #         self.idx = idx


class SizeMeta(type):

    def __call__(cls, *args, **kwargs):

        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args, **kwargs)
        return obj

    def __getitem__(cls, idx: typing.Union[int, tuple]):
        if isinstance(idx, tuple):
            if len(idx) > 2:
                raise KeyError(f'Index size must be less than or equal to 2 not {len(idx)}')
            return cls(idx[0], idx[1])
        return cls(idx)


class sz(object, metaclass=SizeMeta):

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


class LabelSet:
    
    def __init__(self, labels: typing.Iterable[str]=None):
        labels = labels or []
        self._labels = set(*labels)
    
    def __contains__(self, key):
        return key in self._labels


@dataclass
class Info:

    name: str=''
    labels: LabelSet=field(default_factory=LabelSet)
    annotation: str=''

    def __post_init__(self):
        if isinstance(self.labels, typing.List):
            self.labels = LabelSet(self.labels)


class OpFactory(ABC):

    def __init__(self, info: Info=None):
        self._info = info or Info()

    @abstractmethod
    def produce(self, in_size: torch.Size, **kwargs) -> typing.Tuple[nn.Module, torch.Size]:
        raise NotImplementedError
    
    @abstractmethod
    def produce_nodes(self, in_size: torch.Size, **kwargs) -> typing.Iterator[Node]:
        raise NotImplementedError

    @abstractmethod
    def to(self, **kwargs):
        raise NotImplementedError
    
    def alias(self, **kwargs):
        return self.to(**{k: var(v) for k, v in kwargs.items()})

    def info(self):
        return self._info


OpFactory.__call__ = OpFactory.to


class Sequence(OpFactory):

    def __init__(self, op_factories: typing.List[OpFactory], info: Info=None):
        super().__init__(info)
        self._op_factories = op_factories
    
    def add(self, op_factory: OpFactory, position: int=None):

        if position is None:
            self._op_factories.append(op_factory)
        else:
            self._op_factories.insert(op_factory, position)

    def __lshift__(self, other: OpFactory):
        if isinstance(other, Sequence):
            return Sequence(self._op_factories + other._op_factories)
        return Sequence(self._op_factories + [other])
    
    def produce(self, in_: typing.List[torch.Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        sequential = nn.Sequential()

        for i, factory in enumerate(self._op_factories):
            module, in_ = factory.produce(in_, **kwargs)
            sequential.add_module(str(i), module)
    
        return sequential, in_

    def produce_nodes(self, in_: typing.List[Port], **kwargs) -> typing.Iterator[Node]:
        for factory in self._op_factories:
            for node in factory.produce_nodes(in_, **kwargs):
                in_ = node.ports  
                yield node
    
    def to(self, **kwargs):
        return Sequence(
            [factory.to(**kwargs) for factory in self._op_factories]
        )
    
    @property
    def info(self):
        return self._info


@abstractmethod
def _lshift(self, op_factory) -> Sequence:
    raise NotImplementedError

OpFactory.__lshift__ = _lshift


class _ArgMap(ABC):

    @singledispatchmethod
    def _remap_arg(self, val, kwargs):
        return val

    @_remap_arg.register
    def _(self, val: var, kwargs):
        return val.to(**kwargs)

    @singledispatchmethod
    def _lookup_arg(self, val, in_size: torch.Size, kwargs):
        return val

    @_lookup_arg.register
    def _(self, val: var, in_size: torch.Size, kwargs):
        return val.to(**kwargs)

    @_lookup_arg.register
    def _(self, val: sz, in_size: torch.Size, kwargs):
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
    def to(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def produce(self, in_: typing.List[Port], **kwargs):
        raise NotImplementedError

    @abstractproperty
    def module(self):
        raise NotImplementedError


class Out(ABC):

    @abstractmethod
    def to(self, **kwargs):
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
        self, module: BaseMod, out: Out=None, info: Info=None
    ):
        super().__init__(info)
        self._out = to_out(out)
        self._mod = module
    
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
        name = self._info.name or type(module).__name__

        yield OpNode(
            name, module, in_, self._out.produce(module, in_), self._info.labels,
            self._info.annotation
        )

    def to(self, **kwargs):
        mod = self._mod.to(**kwargs)
        return BasicOp(
            mod, self._out.to(**kwargs), self._info
        )

    
ModType = typing.Union[typing.Type[nn.Module], var]
ModInstance = typing.Union[nn.Module, var]


def kwarg_pop(key, kwargs):

    result = None
    if '_out' in kwargs:
        result = kwargs.get('_out')
        del kwargs['_out']
    
    return result


class NNMod(object):

    def __init__(self, nnmodule: typing.Type[nn.Module]):

        self._nnmodule = nnmodule

    def __call__(self, *args, **kwargs) -> BasicOp:

        out_ = to_out(kwarg_pop('_out', kwargs))
        info = kwarg_pop('_info', kwargs)
        
        return BasicOp(ModFactory(self._nnmodule, *args, **kwargs), out_, info)


class OpMod(object):

    def __init__(self, mod):

        self._mod = mod

    def __getattribute__(self, __name: str) -> NNMod:
        mod = super().__getattribute__('_mod')
        nnmodule = getattr(mod, __name)

        if not issubclass(nnmodule, nn.Module):
            raise AttributeError(f'Attribute {__name} is not a valid nn.Module')

        return NNMod(nnmodule)


class ModFactory(BaseMod):

    def __init__(self, module: ModType, *args, **kwargs):

        self._module = module
        self._args = ArgSet(*args, **kwargs)

    def to(self, **kwargs):
        args = self._args.remap(kwargs)
        module = self._module.to(**kwargs) if isinstance(self._module, var) else self._module
        return ModFactory(module, *args.args, *args.kwargs)

    @singledispatchmethod
    def produce(self, in_: typing.List[torch.Size], **kwargs):
        
        module = self._module.to(**kwargs) if isinstance(self._module, var) else self._module
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

    def op(self, out_: Out=None, info: Info=None) -> BasicOp:
        return BasicOp(self, to_out(out_), info)


class Instance(BaseMod):

    def __init__(self, module: ModInstance):

        self._module = module

    def to(self, **kwargs):
        module = self._module.to(**kwargs) if isinstance(self._module, var) else self._module
        return module

    def produce(self, in_: typing.List[torch.Size], **kwargs):
        
        return self._module.to(**kwargs) if isinstance(self._module, var) else self._module
    
    @property
    def module(self):
        return self._module

    def op(self, out_: Out=None, name: str=None, labels: typing.List[str]=None, annotation: str=None) -> BasicOp:
        return BasicOp(self, to_out(out_), name, labels, annotation)


@singledispatch
def factory(mod: ModType, *args, **kwargs):
    return ModFactory(mod, *args, *kwargs)

@factory.register
def _(mod: str, *args, **kwargs):
    return ModFactory(var(mod), *args, *kwargs)

@singledispatch
def instance(mod: ModInstance):
    return Instance(mod)

@instance.register
def _(mod: str):
    return Instance(var(mod))

# TODO: Need to fix this

class ListOut(Out):

    def __init__(self, sizes: typing.List):

        # TODO: Take care of the case that multiple lists can be output
        if len(sizes) == 0 or not isinstance(sizes[0], list):
            sizes = [sizes]
        
        self._sizes = [Args(*size) for size in sizes]

    def to(self, **kwargs):
        
        sizes = [list(size.remap(kwargs).items) for size in self._sizes]
        return ListOut(sizes)

    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs): 
        return [
            torch.Size(size.lookup(in_size, kwargs).items) 
            for size in self._sizes
        ]


class SizeOut(Out):

    def __init__(self, size: torch.Size):
        
        self._size =size
    
    def to(self, **kwargs):
        return SizeOut(self._size)

    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs): 
        return self._size


class NullOut(Out):

    def __init__(self):
        pass

    def to(self, **kwargs):
        return NullOut()

    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):         
        return in_size


class FuncOut(Out):

    def __init__(self, f: typing.Callable[[nn.Module, torch.Size, typing.Dict], torch.Size]):
        
        self._f = f

    def to(self, **kwargs):
        return FuncOut(self._f)

    def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):         
        return FuncOut(mod, in_size, kwargs)


def _func_type():
    pass

_func_type = type(_func_type)


@singledispatch
def to_out(out_=None):
    if out_ is not None:
        raise ValueError(f'Argument out_ is not a valid type {type(out_)}')
    return NullOut()

@to_out.register
def _(out_: list):
    return ListOut(out_)


@to_out.register
def _(out_: _func_type):
    return FuncOut(out_)


@to_out.register
def _(out_: torch.Size):
    return SizeOut(out_)


@to_out.register
def _(out_: Out):
    return out_


class Override(OpFactory):

    pass


def over(factory: OpFactory):

    pass


class DivergeFactory(OpFactory):

    def __init__(
        self, op_factories: typing.List[OpFactory], info: Info=None
    ):
        super().__init__(info)
        self._op_factories = op_factories
    
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
    
    def to(self, **kwargs):
        return DivergeFactory(
            [op_factory.to(**kwargs) for op_factory in self._op_factories],
            self._info
        )


diverge = DivergeFactory


class ParallelFactory(OpFactory):

    def __init__(
        self, op_factories: typing.List[OpFactory], info: Info=None
    ):
        super().__init__(info)
        self._op_factories = op_factories
    
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

    def to(self, **kwargs):
        return ParallelFactory(
            [op_factory.to(**kwargs) for op_factory in self._op_factories],
            self._info
        )


parallel = ParallelFactory


class Chain(OpFactory):
    # need to allow for 
    def __init__(
        self, op_factory: OpFactory, attributes: typing.Union[var, typing.List[Kwargs]],
        info: Info=None
    ):
        super().__init__(info)
        self._op_factory = op_factory
        self._attributes = attributes
    
    def __lshift__(self, other) -> Sequence:
        return Sequence([self, other])

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Size):
            in_ = [in_]
        attributes = self._attributes
        if isinstance(attributes, var):
            attributes = self._attributes.to(**kwargs)
        
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
        if isinstance(attributes, var):
            attributes = self._attributes.to(**kwargs)
        
        for attribute in self._attributes:
            for node in self._op_factory.produce_nodes(in_, **attribute.items):
                yield node
            # Use produce to get the next in_size
            _, in_ = self._op_factory.produce(in_, **attribute.items)

    def to(self, **kwargs):
        attributes = self._attributes
        if isinstance(self._attributes, var):
            attributes = self._attributes.to(**kwargs)
        return Chain(
            self._op_factory.to(kwargs), attributes,
            self._info
        )


chain = Chain



class InFactory(ABC):

    def produce(self, **kwargs) -> Node:
        pass


class TensorInFactory(InFactory):

    def __init__(self, size: torch.Size, default, call_default: bool=False, device: str='cpu', info: Info=None):
        
        self._default = default
        self._sz = size
        self._call_default = call_default
        self._device = device
        self._info = info

    def produce(self, **kwargs) -> Node:

        default = self._default(*self._sz, device=self._device) if self._call_default else self._default    
        return In.from_tensor(self._info.name, self._type_, default, self._info.labels, self._info.annotation)

tensor_in = TensorInFactory


class ScalarInFactory(InFactory):

    def __init__(self, type_: typing.Type, default, call_default: bool=False, info: Info=None):

        self._type_ = type_
        self._default = default
        self._call_default = call_default
        self._info = info or Info()

    def produce(self, **kwargs) -> Node:

        default = self._default() if self._call_default else self._default    
        return In.from_scalar(self._info.name, self._type_, default, self._info.labels, self._info.annotation)

scalar_in = ScalarInFactory


class ParameterFactory(InFactory):

    def __init__(self, size: torch.Size, default, call_default: bool=False, device: str='cpu', info: Info=None):
        
        self._default = default
        self._sz = size
        self._call_default = call_default
        self._device = device
        self._info = info

    def produce(self, **kwargs) -> Node:

        return Parameter(self._info.name, self._sz, self._reset_func, self._info.labels, self._info.annotation)
        
param_in = ParameterFactory



class BuildMultitap(object):

    def __init__(self, builder, multitap: Multitap):
        
        self._builder: NetBuilder = builder
        self._multitap = multitap

    def __lshift__(self, op_factory: OpFactory):
        
        multitap = self._multitap
        for node in op_factory.produce_nodes(self._multitap.sizes):
            multitap = self._builder.add_node(node)
        return BuildMultitap(self._builder, multitap)


class NetBuilder(object):
    """
    Builder class with convenience methods for building networks
    - Handles adding nodes to the network to simplify that process
    - Do not need to specify the incoming ports if the incoming ports
    -  are the outputs of the previously added node
    - Can set base labels that will apply to all nodes added to the network
    -  that use the convenience methods (not add_node)
    """

    # later add in 

    def __init__(self):

        self._net = Network()

    def __getitem__(self, keys: list):
        
        node_set: NodeSet = self._net[keys]
        return BuildMultitap(self, node_set.ports)

    def add_ins(self, in_: typing.List[InFactory]):
        return self._net.add_node(in_)

    def add_in(self, in_: InFactory):
        return self._net.add_node(in_)

    def add_node(self, node: Node):
        return self._net.add_node(node)

    def __lshift__(self, in_: InFactory):
        return BuildMultitap(self, self._net.add_node(in_.produce()))

    @property
    def net(self):
        return self._net

    def set_default_interface(self, ins: typing.List[typing.Union[Port, str]], outs: typing.List[typing.Union[Port, str]]):
        self._net.set_default_interface(
            ins, outs
        )

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

#     def to(self, **kwargs):
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

#     def to(self, **kwargs):
#         var_module = self._var_module.to(**kwargs)

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
#         module = self._module.to(**kwargs) if isinstance(self._module, Var) else self._module
#         module = module(*my_args, **my_kwargs)
#         return module

# mod = Mod
