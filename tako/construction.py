 
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from functools import singledispatch, singledispatchmethod
from os import path
from typing import Any, Counter, Iterator, TypeVar
import typing
from numpy import isin
import torch
from torch import nn
from torch import Size
from .networks import In, ModRef, Multitap, Network, Node, NodeSet, OpNode, Parameter, Port, Out
from .modules import Multi, Multi, Diverge
from functools import wraps


T = TypeVar('T')


class arg(object):

    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self):
        return self._name

    def to(self, **kwargs):
        return kwargs.get(self._name, self)


class __arg(object):

    def __getattribute__(self, __name: str) -> arg:
        return arg(__name)


arg_ = __arg()


def to_multitap(f):

    @wraps(f)
    def produce_nodes(self, in_, **kwargs):
        if isinstance(in_, Port):
            in_ = Multitap([in_])
        
        elif not isinstance(in_, Multitap):
            # assume it is a list
            in_ = Multitap([*in_])

        return f(self, in_, **kwargs)
    return produce_nodes


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


class argf(object):

    def __init__(self, args, f):

        self._f = f
        self._args = args
    
    def to(self, **kw):
        args = []
        for a in self._args:
            if isinstance(a, arg):
                args.append(a.to(kw))
            else:
                args.append(a)
        return argf(args, self._f)
    
    def process(self, sizes: typing.List[torch.Size], **kwargs):
        _args = []
        for a in self._args:
            if isinstance(a, sz):
                _args.append(a.process(sizes))
            elif isinstance(a, arg):
                _args.append(a.to(kwargs))
            else:
                _args.append(a)
        return self._f(*_args)


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
    fix: bool=False

    def __post_init__(self):
        if isinstance(self.labels, typing.List):
            self.labels = LabelSet(self.labels)
    
    def spawn(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        return Info(
            name if name is not None else self.name,
            LabelSet(labels) if labels is not None else self.labels,
            annotation if annotation is not None else self.annotation, 
            fix if fix is not None else self.fix
        )


class NetFactory(ABC):

    def __init__(self, info: Info=None):
        self._info = info or Info()

    @abstractmethod
    def produce(self, in_size: torch.Size, **kwargs) -> typing.Tuple[nn.Module, torch.Size]:
        raise NotImplementedError
    
    @abstractmethod
    def produce_nodes(self, in_: Multitap, **kwargs) -> typing.Iterator[Node]:
        raise NotImplementedError

    @abstractmethod
    def to(self, **kwargs):
        raise NotImplementedError
    
    def alias(self, **kwargs):
        return self.to(**{k: arg(v) for k, v in kwargs.items()})

    @property
    def info(self):
        return self._info

    def info_(self, name: str='', labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        pass


NetFactory.__call__ = NetFactory.to


class SequenceFactory(NetFactory):

    def __init__(self, op_factories: typing.List[NetFactory], info: Info=None):
        super().__init__(info)
        self._op_factories = op_factories
    
    def add(self, op_factory: NetFactory, position: int=None):

        if position is None:
            self._op_factories.append(op_factory)
        else:
            self._op_factories.insert(op_factory, position)

    def __lshift__(self, other: NetFactory):
        if isinstance(other, SequenceFactory):
            return SequenceFactory(self._op_factories + other._op_factories)
        return SequenceFactory(self._op_factories + [other])
    
    def produce(self, in_: typing.List[Out], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Out):
            in_ = [in_]
        sequential = nn.Sequential()

        for i, factory in enumerate(self._op_factories):
            module, in_ = factory.produce(in_, **kwargs)
            sequential.add_module(str(i), module)
    
        return sequential, in_

    @to_multitap
    def produce_nodes(self, in_: Multitap, **kwargs) -> typing.Iterator[Node]:
        for factory in self._op_factories:
            for node in factory.produce_nodes(in_, **kwargs):
                in_ = Multitap(node.ports)  
                yield node
    
    def to(self, **kwargs):
        return SequenceFactory(
            [factory.to(**kwargs) for factory in self._op_factories]
        )
    
    @property
    def info(self):
        return self._info

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):        
        return SequenceFactory(self._op_factories, self._info.spawn(name, labels, annotation, fix))


@abstractmethod
def _lshift(self, net_factory) -> SequenceFactory:
    raise NotImplementedError

NetFactory.__lshift__ = _lshift


class _ArgMap(ABC):

    @singledispatchmethod
    def _remap_arg(self, val, kwargs):
        return val

    @_remap_arg.register
    def _(self, val: arg, kwargs):
        return val.to(**kwargs)

    @_remap_arg.register
    def _(self, val: argf, kwargs):
        return val.to(**kwargs)

    @singledispatchmethod
    def _lookup_arg(self, val, in_size: torch.Size, kwargs):
        return val

    @_lookup_arg.register
    def _(self, val: arg, in_size: torch.Size, kwargs):
        return val.to(**kwargs)

    @_lookup_arg.register
    def _(self, val: sz, in_size: torch.Size, kwargs):
        return val.process(in_size)

    @_lookup_arg.register
    def _(self, val: argf, in_size: torch.Size, kwargs):
        return val.process(in_size, kwargs)

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

    def remap_keys(self, kwargs):

        remapped = {}
        for x, y in kwargs.items():
            if x in self._kwargs:
                y: arg = y
                remapped[y.name] = self._kwargs[x] 

        return Kwargs(**remapped)

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
    
    def lookup(self, sizes: typing.List[torch.Size], kwargs):

        return ArgSet(*self._args.lookup(sizes, kwargs).items, **self._kwargs.lookup(sizes, kwargs).items)
    
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


def compute_out_sizes(mod, in_: typing.List[Port]) -> typing.Tuple[torch.Size]:

    in_tensors = []

    # get in_tensors
    for in_i in in_:
        size = [*in_i.size]
        if size[0] == -1:
            size[0] = 1
        x = torch.zeros(*size, dtype=in_i.dtype)
        in_tensors.append(x)
    y = mod(*in_tensors)
    if isinstance(y, torch.Tensor):
        y = [y]
    
    outs = []
    for y_i in y:
        size = [*y_i.size()]
        print(y_i.size())
        size[0] = -1
        outs.append(Out(torch.Size(size), y_i.dtype))
    return outs


class OpFactory(NetFactory):

    def __init__(
        self, module: BaseMod, info: Info=None
    ):
        super().__init__(info)
        self._mod = module
    
    def __lshift__(self, other) -> SequenceFactory:
        return SequenceFactory([self, other])

    def produce(self, in_: typing.List[Out], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Out):
            in_ = [in_]

        module = self._mod.produce([in_i.size for in_i in in_], **kwargs)
        return module, compute_out_sizes(module, in_)
    
    @to_multitap
    def produce_nodes(self, in_: Multitap, **kwargs) -> typing.Iterator[Node]:
        
        module = self._mod.produce([in_i.size for in_i in in_], **kwargs)
        name = self._info.name if self._info.name != '' else type(module).__name__

        outs = compute_out_sizes(module, in_)
        op_node = OpNode(
            name, module, in_, outs, self._info.labels,
            self._info.annotation
        )

        yield op_node

    def to(self, **kwargs):
        mod = self._mod.to(**kwargs)
        return OpFactory(
            mod, self._info
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        
        return OpFactory(self._mod, self._info.spawn(name, labels, annotation, fix))

    
ModType = typing.Union[typing.Type[nn.Module], arg]
ModInstance = typing.Union[nn.Module, arg]


def kwarg_pop(key, kwargs):

    result = None
    if '_out' in kwargs:
        result = kwargs.get('_out')
        del kwargs['_out']
    
    return result


class NNMod(object):

    def __init__(self, nnmodule: typing.Type[nn.Module]):

        self._nnmodule = nnmodule

    def __call__(self, *args, **kwargs) -> OpFactory:

        # out_ = to_out(kwarg_pop('_out', kwargs))
        info = kwarg_pop('_info', kwargs)
        
        return OpFactory(ModFactory(self._nnmodule, *args, **kwargs), info)


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
        module = self._module.to(**kwargs) if isinstance(self._module, arg) else self._module
        return ModFactory(module, *args.args, *args.kwargs)

    @singledispatchmethod
    def produce(self, in_: typing.List[torch.Size], **kwargs):
        
        if isinstance(in_, torch.Size):
            in_ = [in_]
        module = self._module.to(**kwargs) if isinstance(self._module, arg) else self._module

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

    def op(self, info: Info=None) -> OpFactory:
        return OpFactory(self, info)


class Instance(BaseMod):

    def __init__(self, module: ModInstance):

        self._module = module

    def to(self, **kwargs):
        module = self._module.to(**kwargs) if isinstance(self._module, arg) else self._module
        return module

    def produce(self, in_: typing.List[torch.Size], **kwargs):
        
        return self._module.to(**kwargs) if isinstance(self._module, arg) else self._module
    
    @property
    def module(self):
        return self._module

    def op(self, name: str=None, labels: typing.List[str]=None, annotation: str=None) -> OpFactory:
        return OpFactory(self, Info(name, labels, annotation))


@singledispatch
def factory(mod: ModType, *args, **kwargs):
    return ModFactory(mod, *args, *kwargs)

@factory.register
def _(mod: str, *args, **kwargs):
    return ModFactory(arg(mod), *args, *kwargs)

@singledispatch
def instance(mod: ModInstance):
    return Instance(mod)

@instance.register
def _(mod: str):
    return Instance(arg(mod))


class DivergeFactory(NetFactory):

    def __init__(
        self, op_factories: typing.List[NetFactory], info: Info=None
    ):
        super().__init__(info)
        self._op_factories = op_factories
    
    def __lshift__(self, other) -> SequenceFactory:
        return SequenceFactory([self, other])

    def produce(self, in_: typing.List[Out], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Out):
            in_ = [in_]
        mods = []
        outs = []
        for in_i, op_factory in zip(in_, self._op_factories):
            mod, out = op_factory.produce(in_i, **kwargs)
            mods.append(mod)
            outs.extend(out)
        mod = Diverge(mods)
        return mod, outs
    
    @to_multitap
    def produce_nodes(self, in_: Multitap, **kwargs) -> typing.Iterator[Node]:
        
        if isinstance(in_, Port):
            in_ = [in_]
        
        for in_i, op_factory in zip(in_, self._op_factories):
            for node in op_factory.produce_nodes(Multitap([in_i]), **kwargs):
                yield node
    
    def to(self, **kwargs):
        return DivergeFactory(
            [op_factory.to(**kwargs) for op_factory in self._op_factories],
            self._info
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        
        return DivergeFactory(self._op_factories, self._info.spawn(name, labels, annotation, fix))

diverge = DivergeFactory


class MultiFactory(NetFactory):

    def __init__(
        self, op_factories: typing.List[NetFactory], info: Info=None
    ):
        super().__init__(info)
        self._op_factories = op_factories
    
    def __lshift__(self, other) -> SequenceFactory:
        return SequenceFactory([self, other])

    def produce(self, in_: typing.List[Out], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Out):
            in_ = [in_]
        mods = []
        outs = []
        for op_factory in self._op_factories:
            mod, out = op_factory.produce(in_, **kwargs)
            mods.append(mod)
            outs.extend(out)
        mod = Multi(mods)
        return mod, outs
    
    @to_multitap
    def produce_nodes(self, in_: Multitap, **kwargs) -> typing.Iterator[Node]:
        
        for op_factory in self._op_factories:
            for node in op_factory.produce_nodes(in_, **kwargs):
                yield node

    def to(self, **kwargs):
        return MultiFactory(
            [op_factory.to(**kwargs) for op_factory in self._op_factories],
            self._info
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        
        return MultiFactory(self._op_factories, self._info.spawn(name, labels, annotation, fix))


multi = MultiFactory


class Chain(NetFactory):
    def __init__(
        self, op_factory: NetFactory, attributes: typing.Union[arg, typing.List[Kwargs]],
        info: Info=None
    ):
        super().__init__(info)
        self._op_factory = op_factory
        self._attributes = attributes
    
    def __lshift__(self, other) -> SequenceFactory:
        return SequenceFactory([self, other])

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Size):
            in_ = [in_]
        attributes = self._attributes
        if isinstance(attributes, arg):
            attributes = self._attributes.to(**kwargs)
        
        mods = []
        out = in_
        for attribute in self._attributes:
            mod, out = self._op_factory.produce(out, **attribute.items)
            mods.append(mod)
        return nn.Sequential(*mods), out
    
    @to_multitap
    def produce_nodes(self, in_: Multitap, **kwargs) -> typing.Iterator[Node]:
        
        attributes = self._attributes
        if isinstance(attributes, arg):
            attributes = self._attributes.to(**kwargs)
        
        for attribute in self._attributes:
            for node in self._op_factory.produce_nodes(in_, **attribute.items):
                yield node
                in_ = Multitap(node.ports)

    def to(self, **kwargs):
        attributes = self._attributes
        if isinstance(self._attributes, arg):
            attributes = self._attributes.to(**kwargs)
        
        to_attributes = []
        for attribute in attributes:
            to_attributes.append(attribute.remap_keys(kwargs))
            
        return Chain(
            self._op_factory.to(**kwargs), to_attributes,
            self._info
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        
        return Chain(self._op_factory, self._attributes, self._info.spawn(name, labels, annotation, fix))

chain = Chain


class InFactory(ABC):

    def coalesce_name(self, name):
        return name if name not in ('', None) else type(self).__name__

    def produce(self, **kwargs) -> Node:
        pass


class TensorInFactory(InFactory):

    def __init__(
        self, size: typing.Union[torch.Size, typing.Iterable], dtype: torch.dtype, 
        default, call_default: bool=False, device: str='cpu', info: Info=None
    ):
        
        self._default = default
        if not isinstance(size, torch.Size):
            size = torch.Size(size)

        self._dtype = dtype
        self._size = size
        self._call_default = call_default
        self._device = device
        self._info = info or Info(name='Tensor')

    def produce(self) -> In:

        size = [*self._size]
        if self._size[0] == -1:
            size[0] = 1

        default = self._default(
            *size, device=self._device
        ) if self._call_default else self._default  
        
        return In.from_tensor(
            self._info.name, self._size, self._dtype, default, 
            self._info.labels, self._info.annotation
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        
        return TensorInFactory(self._size, self._dtype, self._default, self._call_default, self._device, self._info.spawn(name, labels, annotation, fix))

tensor_in = TensorInFactory


class ScalarInFactory(InFactory):

    def __init__(self, type_: typing.Type, default, call_default: bool=False, info: Info=None):

        self._type_ = type_
        self._default = default
        self._call_default = call_default
        self._info = info or Info(name='Scalar')

    def produce(self, **kwargs) -> Node:

        default = self._default() if self._call_default else self._default    
        return In.from_scalar(self._info.name, self._type_, default, self._info.labels, self._info.annotation)

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        
        return ScalarInFactory(self._type_, self._default, self._call_default, self._info.spawn(name, labels, annotation, fix))


scalar_in = ScalarInFactory


class ParameterFactory(InFactory):

    def __init__(self, size: torch.Size, dtype: torch.dtype, reset_func, device: str='cpu', info: Info=None):
        
        self._reset_func = reset_func
        self._sz = size
        self._device = device
        self._info = info or Info(name='Param')
        self._dtype = dtype

    def produce(self, **kwargs) -> Node:

        return Parameter(self._info.name, self._sz, self._dtype, self._reset_func, self._info.labels, self._info.annotation)
        
    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        
        return ParameterFactory(self._sz, self._dtype, self._reset_func, self._device, self._info.spawn(name, labels, annotation, fix))

param_in = ParameterFactory


class BuildMultitap(object):

    def __init__(self, builder, multitap: Multitap):
        
        if isinstance(multitap, list) or isinstance(multitap, tuple):
            multitap = Multitap(multitap)
        self._builder: NetBuilder = builder
        self._multitap = multitap
    
    # TODO: consider whether to keep this
    @property
    def multitap(self):
        return self._multitap
    
    @property
    def ports(self):
        return [*self._multitap.ports]
    
    @singledispatchmethod
    def __getitem__(self, idx: typing.Iterable) -> Multitap:
        return self._multitap[idx]

    @__getitem__.register
    def _(self, idx: int) -> Port:
        return self._multitap[idx]
    
    def __iter__(self) -> typing.Iterator:
        for port in self._multitap:
            yield port

    def __lshift__(self, net_factory: NetFactory):
        
        multitap = self._multitap
        for node in net_factory.produce_nodes(self._multitap.ports):
            multitap = Multitap(self._builder.add_node(node))
        return BuildMultitap(self._builder, multitap)


def to_multitap(**mapping):
    ports = []

    for k, v in mapping.items():
        if isinstance(v, str):
            ports.append(Port(ModRef(k)))
        elif isinstance(v, Port):
            ports.append(v)
        elif isinstance(v, Multitap):
            ports.extend(v.ports)
        elif isinstance(v, BuildMultitap):
            ports.extend(v.ports)
        else:
            # TODO: make this more ammenable to extension
            raise ValueError("Cannot process mapping")
    return Multitap(ports)


class NetBuilder(object):
    """
    Builder class with convenience methods for building networks
    - Handles adding nodes to the network to simplify that process
    - Do not need to specify the incoming ports if the incoming ports
    -  are the outputs of the previously added node
    - Can set base labels that will apply to all nodes added to the network
    -  that use the convenience methods (not add_node)
    """

    def __init__(self):
        self._net = Network()
        self._names = Counter()

    def __getitem__(self, keys: list):
        multitap: Multitap = to_multitap(keys)
        return BuildMultitap(self, multitap.ports)

    def add_ins(self, in_: typing.List[InFactory]):
        ports = []
        for in_i in in_:
            ports.extend(self.add_in(in_i))
        return ports

    def add_in(self, in_: InFactory):
        node = in_.produce()
        return self.add_node(node)

    def add_node(self, node: Node):
        self._names.update([node.name])
        if self._names[node.name] > 1:
            node.name = f'{node.name}_{self._names[node.name]}'
        return self._net.add_node(node)

    def __lshift__(self, in_: InFactory):
        ports = self.add_in(in_)
        return BuildMultitap(self, ports)

    @property
    def net(self):
        return self._net

    def set_default_interface(self, ins: typing.List[typing.Union[Port, str]], outs: typing.List[typing.Union[Port, str]]):
        self._net.set_default_interface(
            ins, outs
        )
    
    # TODO: Figure out how to implement
    # add in port mapping
    # def build_machine(self, name: str, *learner_mixins: typing.Type[MachineComponent], **name_map):

    #     class _(*learner_mixins):
    #         __qualname__ = name

    #     return _(self.net)


# class Out(ABC):

#     @abstractmethod
#     def to(self, **kwargs):
#         raise NotImplementedError

#     @abstractmethod
#     def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):
#         raise NotImplementedError


# class ListOut(Out):

#     def __init__(self, sizes: typing.List):

#         # TODO: Take care of the case that multiple lists can be output
#         if len(sizes) == 0 or not isinstance(sizes[0], list):
#             sizes = [sizes]
        
#         self._sizes = [Args(*size) for size in sizes]

#     def to(self, **kwargs):
#         sizes = [list(size.remap(kwargs).items) for size in self._sizes]
#         return ListOut(sizes)

#     def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs): 
#         return [
#             torch.Size(size.lookup(in_size, kwargs).items) 
#             for size in self._sizes
#         ]


# class SizeOut(Out):

#     def __init__(self, size: torch.Size):
        
#         self._size = size
    
#     def to(self, **kwargs):
#         return SizeOut(self._size)

#     def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs): 
#         return self._size


# class NullOut(Out):

#     def __init__(self):
#         pass

#     def to(self, **kwargs):
#         return NullOut()

#     def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):         
#         return in_size


# class FuncOut(Out):

#     def __init__(self, f: typing.Callable[[nn.Module, torch.Size, typing.Dict], torch.Size]):
        
#         self._f = f

#     def to(self, **kwargs):
#         return FuncOut(self._f)

#     def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):         
#         return self._f(mod, in_size, kwargs)


# class ArgfOut(Out):

#     def __init__(self, f: argf):
#         self._f: argf = f

#     def to(self, **kwargs):
#         return ArgfOut(self._f.to(kwargs))

#     def produce(self, mod: nn.Module, in_size: torch.Size, **kwargs):         
#         return self._f.process(in_size, kwargs)
        

# def _func_type():
#     pass

# _func_type = type(_func_type)


# @singledispatch
# def to_out(out_=None):
#     if out_ is not None:
#         raise ValueError(f'Argument out_ is not a valid type {type(out_)}')
#     return NullOut()

# @to_out.register
# def _(out_: list):
#     return ListOut(out_)


# @to_out.register
# def _(out_: _func_type):
#     return FuncOut(out_)


# @to_out.register
# def _(out_: torch.Size):
#     return SizeOut(out_)


# @to_out.register
# def _(out_: Out):
#     return out_


# @to_out.register
# def _(out_: argf):
#     return ArgfOut(out_)

# define interface that must be defined in learn, test, machine mixins
# 

# learner = builder.build_learner('X', SGDLearn, StandardTest, [Regressor])

# creates your learner object with the correct class
# checks the interface of the network!


# mod(nn.Linear, 2, 3).op(out=(-1, Sz(1))
# mod(nn.Linear, 2, 3).op()
# linear = mod(nn.Linear, Var('x'), Var('y')).op(out=(-1, Sz(1))) << mod(nn.BatchNorm(Sz(1))) << mod(nn.Sigmoid)

# sequence = linear(x=2, y=3) << linear(x=3, y=4)
# 

# mod(nn.Conv2d, kw=2, kh=kl, stride=2, stride=3 ).op(fc)

