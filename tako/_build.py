"""
Modules for building a network

Arguments:
arg - Evaluate arguments to a constructor 
sz - Input size arguments to a constructor
argf - Evaluate arguments to a constructor after building

Node/Network building:
In - Build input nodes to a network
NetFactory - Build nodes to a network
  they can also build a standard torch neural network with
  Sequential
NetBuilder - Building a neural network
Namer - Name a node
"""
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from functools import singledispatch, singledispatchmethod
from os import path
from typing import Any, Counter, TypeVar
import typing
import torch
from torch import nn
from torch import Size
from ._networks import (
    In, Meta, InScalar, InTensor, Multitap, 
    Network, NetworkInterface, Node, NodePort, NodeSet, OpNode, Port, Out
)
from ._modules import (
    Multi, Multi, Diverge
)
from functools import wraps


T = TypeVar('T')


class arg(object):

    @abstractmethod
    def to(self, **kwargs):
        """Update the value of the argument

        Returns:
            Any: value specified in kwargs if it is there else the arg
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, sizes: typing.List[torch.Size], kwargs: dict) -> int:
        raise NotImplementedError


class argv(arg):
    """Argument for a NetFactory/ModFactory
    Use to add a value that is undefined 

    x = arg('x').to(1)
    -> x == 1
    """

    def __init__(self, name: str):
        """initializer

        Args:
            name (str): Name fo the argument
        """
        self._name = name
    
    @property
    def name(self):
        return self._name

    def to(self, **kwargs):
        """Update the value of the argument

        Returns:
            Any: value specified in kwargs if it is there else the arg
        """
        return kwargs.get(self._name, self)

    def process(self, sizes: typing.List[torch.Size], kwargs: dict=None):
        kwargs = kwargs or {}
        return self.to(**kwargs)


class __arg(object):
    """Convenience module for creating an arg

    arg = arg_.x
    -> arg == arg("x")
    """

    def __getattribute__(self, __name: str) -> arg:
        return argv(__name)

# an instance of __arg to create an arg
arg_ = __arg()


def to_multitap(f):
    """Decorator to convert args to a multitap

    Args:
        f: produce_nodes() function
    """

    @wraps(f)
    def produce_nodes(self, in_, namer: Namer=None, **kwargs):
        if isinstance(in_, Port):
            in_ = Multitap([in_])
        
        elif not isinstance(in_, Multitap):
            # assume it is a list
            in_ = Multitap([*in_])

        return f(self, in_, namer, **kwargs)
    return produce_nodes


class SizeMeta(type):
    """Use to easily create an sz object with indexing
    """

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


class sz(arg, metaclass=SizeMeta):
    """Reference to the Size of an input
    """

    def __init__(self, dim_idx: int, port_idx: int=None):
        """initializer

        Args:
            dim_idx (int): The dimensional index it refers to
            port_idx (int, optional): The index to the port. Only use if node has multiple ports.
            Defaults to None.
        """
        self._port_idx = port_idx
        self._dim_idx= dim_idx
    
    def to(self, **kwargs):
        return self

    def process(self, sizes: typing.List[torch.Size], kwargs: dict=None) -> int:
        """Replace the reference with size passed in
        """
        kwargs = kwargs or {}

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
    """Argument evaluated by a function
    """

    def __init__(self, f, args):
        """Get arg to a module by passing multiple args to a function

        Usage:

        argf(lambda x, y: x * y, [arg_.features, sz[1]])

        Args:
            f (_type_): _description_
            args (_type_): _description_
        """

        self._f = f
        self._args = args
    
    def to(self, **kw):
        args = []
        for a in self._args:
            if isinstance(a, arg):
                args.append(a.to(**kw))
            else:
                args.append(a)
        return argf(self._f, args)

    def process(self, sizes: typing.List[torch.Size], kwargs: dict=None) -> int:
        _args = []
        kwargs = kwargs or {}
        for a in self._args:
            if isinstance(a, arg):
                _args.append(a.process(sizes, kwargs))
                if isinstance(_args[-1], arg):
                    raise ValueError(f"No value assigned to {_args[-1]}")
            else:
                _args.append(a)
        return self._f(*_args)


def module_name(obj):
    return type(obj).__name__


class Namer(ABC):
    """Names a node
    """

    @abstractmethod
    def name(self, name: str, module=None, default: str='Op') -> Meta:
        """Assign a name to a node
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def _base_name(self, name: str=None, module=None, default: str='Op') -> str:
        """Get base name to assign to a node
        """
        if name:
            return name
        elif module is not None:
            return module_name(module)
        else:
            return default


class NetFactory(ABC):
    """Produces nodes and networks
    """

    def __init__(self, name: str="", meta: Meta=None):
        """initializer

        Args:
            name (str, optional): Name of the operation. Defaults to "".
            meta (Meta, optional): . Defaults to None.
        """
        self._name = name
        self._meta = meta or Meta()

    @abstractmethod
    def produce(self, in_size: torch.Size, **kwargs) -> typing.Tuple[nn.Module, torch.Size]:
        """Produce an nn.Module

        Args:
            in_size (torch.Size): Input size to the network

        Returns:
            typing.Tuple[nn.Module, torch.Size]: Module plus the output size of the network
        """
        raise NotImplementedError
    
    @abstractmethod
    def produce_nodes(self, in_: Multitap, namer: Namer=None, **kwargs) -> typing.Iterator[Node]:
        raise NotImplementedError

    @abstractmethod
    def to(self, **kwargs):
        raise NotImplementedError
    
    def alias(self, **kwargs):
        """Change the names of the args in the factory
        """
        return self.to(**{k: argv(v) for k, v in kwargs.items()})
    
    @property
    def name(self):
        """Name of the factory
        """
        return self._name

    @property
    def meta(self):
        return self._meta

    @abstractmethod
    def info_(self, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        raise NotImplementedError

    @abstractmethod
    def to_ops(self):
        raise NotImplementedError

    def __lshift__(self, other):
        return SequenceFactory([*self.to_ops(), *other.to_ops()])


class FixedNamer(Namer):
    """Always outputs the same name for a given factory
    Useful if you want to control the name used for the node

    namer = FixedNamer()
    print(namer.name("Linear")) # prints "Linear"
    print(namer.name("Linear")) # prints "Linear"
    """

    def name(self, name: str, module=None, default: str='Op') -> Meta:
        return self._base_name(name, module, default)

    def reset(self):
        pass


class CounterNamer(Namer):
    """Appends the count of modules with the name to the name suggested by the factory

    Usage:
    namer = CounterNamer()
    print(namer.name("Linear")) # prints "Linear"
    print(namer.name("Linear")) # prints "Linear 1"

    Args:
        Namer (_type_): _description_
    """

    def __init__(self):
        self._names = Counter()
        self._named: typing.Dict[str, typing.List[str]] = dict()

    def name(self, name: str, module=None, default: str='Op') -> Meta:
        """Decide on a name for the module

        Args:
            name (str): _description_
            module (_type_, optional): _description_. Defaults to None.
            default (str, optional): _description_. Defaults to 'Op'.

        Returns:
            Meta: _description_
        """
        base_name = self._base_name(name, module, default)
        self._names.update([base_name])
        if self._names[base_name] > 1:
            name = f'{base_name}_{self._names[base_name]}'
        else: 
            name = base_name
            self._named[base_name] = []
        self._named[base_name].append(name)
        return name

    def reset(self):
        self._names = Counter()

    def __getitem__(self, key: str):
        return self._named[key]


NetFactory.__call__ = NetFactory.to


class SequenceFactory(NetFactory):
    """Create a sequence of nodes or nn.Sequential
    """

    def __init__(self, op_factories: typing.List[NetFactory], name: str='', meta: Meta=None):
        super().__init__(name, meta)
        self._op_factories = op_factories
    
    def add(self, op_factory: NetFactory, position: int=None):
        """Add op_factory to the sequence

        Args:
            op_factory (NetFactory): Factory to add to the sequence
            position (int, optional): Position of sequence. If none, it appends it. Defaults to None.
        """

        if position is None:
            self._op_factories.append(op_factory)
        else:
            self._op_factories.insert(op_factory, position)

    def to_ops(self):
        return self._op_factories

    def produce(self, in_: typing.List[Out], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Out):
            in_ = [in_]
        sequential = nn.Sequential()

        for i, factory in enumerate(self._op_factories):
            module, in_ = factory.produce(in_, **kwargs)
            sequential.add_module(str(i), module)
    
        return sequential, in_

    @to_multitap
    def produce_nodes(self, in_: Multitap, namer: Namer=None, **kwargs) -> typing.Iterator[Node]:
        namer = namer or FixedNamer()
        for factory in self._op_factories:
            for node in factory.produce_nodes(in_, namer, **kwargs):
                in_ = Multitap(node.ports)
                yield node
    
    def to(self, **kwargs):
        return SequenceFactory(
            [factory.to(**kwargs) for factory in self._op_factories],
            self._name, self._meta
        )
    
    @property
    def meta(self):
        return self._meta

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None):        
        return SequenceFactory(self._op_factories, name or self._name, self._meta.spawn(labels, annotation))


@abstractmethod
def _lshift(self, net_factory) -> SequenceFactory:
    raise NotImplementedError


class _ArgMap(ABC):
    """Store args into a module
    """

    @singledispatchmethod
    def _remap_arg(self, val, kwargs):
        return val

    @_remap_arg.register
    def _(self, val: arg, kwargs):
        return val.to(**kwargs)

    # @_remap_arg.register
    # def _(self, val: argf, kwargs):
    #     return val.to(**kwargs)

    @singledispatchmethod
    def _lookup_arg(self, val, in_size: torch.Size, kwargs):
        return val

    @_lookup_arg.register
    def _(self, val: arg, in_size: torch.Size, kwargs):
        return val.process(in_size, kwargs)

    # @_lookup_arg.register
    # def _(self, val: sz, in_size: torch.Size, kwargs):
    #     return val.process(in_size)

    # @_lookup_arg.register
    # def _(self, val: argf, in_size: torch.Size, kwargs):
    #     return val.process(in_size, kwargs)

    @abstractmethod
    def lookup(self, in_size: torch.Size, kwargs):     
        """Look up the values of args based on the kwargs and
        in_size passed in

        Args:
            kwargs (_type_): Updated values

        Returns:
            arguments
        """
   
        raise NotImplementedError

    @abstractmethod
    def remap(self, kwargs):    
        """Change the value of an argument if in kwargs

        Args:
            kwargs (_type_): Updated values

        Returns:
            cls: A new instance of the class
        """

        raise NotImplementedError


class Kwargs(_ArgMap):
    """Key-value arguments into a factory
    """

    def __init__(self, **kwargs):

        self._kwargs = kwargs

    def lookup(self, in_size: torch.Size, kwargs):
    
        return Kwargs(**{k: self._lookup_arg(v, in_size, kwargs) for k, v in self._kwargs.items()})

    def remap(self, kwargs):
        return Kwargs(
            **{k: self._remap_arg(v, kwargs) 
            for k, v in self._kwargs.items()}
        )

    def remap_keys(self, kwargs):

        remapped = {}
        for x, y in kwargs.items():
            if x in self._kwargs:
                remapped[y.name] = self._kwargs[x] 

        return Kwargs(**remapped)

    @property
    def items(self):
        """The arguments

        Returns:
            dict[str, Any]: Arguments
        """
        return self._kwargs

    @property
    def is_defined(self):
        """        
        Returns:
            bool: whether the Kwargs have been defined.
        """
        for a in self._kwargs.values():
            if isinstance(a, arg):
                return False
        return True

    @property
    def undefined(self):
        """        
        Returns:
            list: The kwargs that have not been defined
        """
        undefined = []
        for a in self._kwargs.values():
            if isinstance(a, arg):
                undefined.append(a)
        return undefined


class Args(_ArgMap):
    """Arguments for a factory
    """

    def __init__(self, *args):
        self._args = args

    def lookup(self, in_size: torch.Size, kwargs):
    
        return Args(*[self._lookup_arg(v, in_size, kwargs) for v in self._args])

    def remap(self, kwargs):

        return Args(*[self._remap_arg(v, kwargs) for v in self._args])
    
    @property
    def items(self):
        """The arguments

        Returns:
            dict[str, Any]: Arguments
        """
        return self._args
    
    @property
    def is_defined(self):
        """        
        Returns:
            bool: whether the args have been defined.
        """
        for a in self._args:
            if isinstance(a, arg):
                return False
        return True

    @property
    def undefined(self):
        """        
        Returns:
            list: The args that have not been defined
        """
        undefined = []
        for a in self._args:
            if isinstance(a, arg):
                undefined.append(a)
        return undefined


class ArgSet(_ArgMap):
    """Container for args and kwargs
    """

    def __init__(self, *args, **kwargs):

        self._args = Args(*args)
        self._kwargs = Kwargs(**kwargs)
    
    def lookup(self, sizes: typing.List[torch.Size], kwargs):

        return ArgSet(*self._args.lookup(sizes, kwargs).items, **self._kwargs.lookup(sizes, kwargs).items)
    
    def remap(self, kwargs):
        return ArgSet(*self._args.remap(kwargs).items, **self._kwargs.remap(kwargs).items)

    @property
    def kwargs(self):
        """The kwargs in the ArgSet
        """
        return self._kwargs.items

    @property
    def args(self):
        """The args in the ArgSet
        """
        return self._args.items
    
    def clone(self):
        return ArgSet(
            self._args.items,
            self._kwargs.items
        )
    
    @property
    def is_defined(self):
        """        
        Returns:
            bool: whether the args have been defined.
        """
        return self._args.is_defined and self._kwargs.is_defined

    @property
    def undefined(self):
        """        
        Returns:
            list: The args that have not been defined
        """
        return self._args.undefined + self._kwargs.undefined


class BaseMod(ABC):
    """Base Module Factory
    """

    @abstractmethod
    def to(self, **kwargs):
        """Convert the args in the module
        """
        raise NotImplementedError

    @abstractmethod
    def produce(self, in_: typing.List[Port], **kwargs):
        """Produce the module
        """
        raise NotImplementedError

    @abstractproperty
    def module(self):
        """Produce the module
        """
        raise NotImplementedError

    @abstractmethod
    def to_ops(self):
        """
        Returns
            ops for the module
        """
        raise NotImplementedError

    def __lshift__(self, other) -> SequenceFactory:
        """
        Returns
            Sequence Factory for the two modules
        """
        return SequenceFactory([*self.to_ops(), *other.to_ops()])


class OpFactory(NetFactory):

    def __init__(
        self, module: BaseMod, name: str="", meta: Meta=None, _out: typing.List[typing.List]=None
    ):
        """A standard operation factory used for a non-container module

        Args:
            module (BaseMod): Factory of the module 
            name (str, optional): Name of the module. Defaults to "".
            meta (Meta, optional): Meta information for the module. Defaults to None.
            _out (typing.List[typing.List], optional): The out size of the module. Defaults to None.
        """
        super().__init__(name, meta)
        self._mod = module
        self._out = _out

    def to_ops(self) -> typing.List[NetFactory]:
        """
        Returns:
            list[NetFactory]: The ops in the module 
        """
        return [self]

    def _in_tensor(self, in_):
        
        in_tensors = []
        for in_i in in_:
            size = [*in_i.size]
            if size[0] == -1:
                size[0] = 1
            x = torch.zeros(*size, dtype=in_i.dtype)
            in_tensors.append(x)
        return in_tensors

    def _out_sizes(self, mod, in_) -> typing.Tuple[torch.Size]:

        y = mod(*self._in_tensor(in_))
        if isinstance(y, torch.Tensor):
            y = [y]
        
        out_guide = self._out or [[] * len(y)]
        outs = []
        for y_i, out_i in zip(y, out_guide):
            size = [*y_i.size()]
            for i, out_el in enumerate(out_i):
                if out_el == -1: size[i] = -1
            else:
                if size: size[0] = -1
            
            outs.append(Out(torch.Size(size), y_i.dtype))
        return outs

    def produce(self, in_: typing.List[Out], **kwargs) -> typing.Tuple[nn.Module, torch.Size]:

        if isinstance(in_, Out):
            in_ = [in_]

        module = self._mod.produce([in_i.size for in_i in in_], **kwargs)
        return module, self._out_sizes(module, in_)
    
    @to_multitap
    def produce_nodes(self, in_: Multitap, namer: Namer=None, **kwargs) -> typing.Iterator[Node]:
        
        namer = namer or FixedNamer()
        module = self._mod.produce([in_i.size for in_i in in_], **kwargs)
        name = namer.name(self._name, module=module)

        outs = self._out_sizes(module, in_)
        if len(outs) == 1:
            outs = outs[0]
        op_node = OpNode(
            name, module, in_, outs, self._meta.spawn()
        )
        yield op_node

    def to(self, **kwargs):
        mod = self._mod.to(**kwargs)
        return OpFactory(
            mod, self._name, self._meta
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None):
        return OpFactory(self._mod, name or self._name, self._meta.spawn(labels, annotation), self._out)

    
ModType = typing.Union[typing.Type[nn.Module], arg]
ModInstance = typing.Union[nn.Module, arg]


class ModFactory(BaseMod):
    """
    Instantiates a module
    """
    def __init__(self, module: ModType, *args, **kwargs):
        """initializer

        Args:
            module (ModType): Module to create
        """

        self._module = module
        self._args = ArgSet(*args, **kwargs)

    def to(self, **kwargs):
        """Remap the args for the module

        Returns:
            ModFactory: Remapped modfactory
        """
        args = self._args.remap(kwargs)
        
        module = self._module.to(**kwargs) if isinstance(self._module, arg) else self._module
        return ModFactory(module, *args.args, **args.kwargs)

    @singledispatchmethod
    def produce(self, in_: typing.List[torch.Size], **kwargs):
        """Produce  module

        Returns:
            nn.Module
        """
        
        if isinstance(in_, torch.Size):
            in_ = [in_]
        module = self._module.to(**kwargs) if isinstance(self._module, arg) else self._module

        args = self._args.lookup(in_, kwargs)
        undefined = args.undefined
        if len(undefined) > 0:
            raise RuntimeError(f"Args {undefined} are not defined in {kwargs}")
        try:
            return module(*args.args, **args.kwargs)
        except Exception as e:
            raise RuntimeError(
                f'Cannot instantiate module {module} with args '
                f'{args.args}, and kwargs {args.kwargs}') from e
    
    @produce.register
    def _(self, in_: torch.Size, **kwargs):
        return self.produce([in_], **kwargs)
    
    @property
    def module(self):
        """
        Returns:
            typing.Type[nn.Module]: Ne module
        """
        return self._module

    @property
    def args(self):
        return self._args.args
    
    @property
    def kwargs(self):
        return self._args.kwargs

    def op(self, name: str='', meta: Meta=None) -> OpFactory:
        """Convert to OpFactory

        Args:
            meta (Meta, optional): Meta information. Defaults to None.

        Returns:
            OpFactory: _description_
        """
        return OpFactory(self, name, meta)

    def to_ops(self) -> typing.List[OpFactory]:
        """Convert to list of NetFactories

        Returns:
            typing.List[OpFactory]
        """
        return [self.op()]


class argmodf(arg):
    """Argument for a NetFactory/ModFactory
    Use to add a value that is undefined 

    x = arg('x').to(1)
    -> x == 1
    """

    def __init__(self, mod_factory: ModFactory):
        """initializer

        Args:
            name (str): Name fo the argument
        """
        self._mod_factory = mod_factory

    def to(self, **kwargs):
        """Update the value of the argument

        Returns:
            Any: value specified in kwargs if it is there else the arg
        """
        return argmodf(self._mod_factory.to(**kwargs))

    def process(self, sizes: typing.List[torch.Size], kwargs: dict=None):
        kwargs = kwargs or {}
        return self._mod_factory.produce(sizes, **kwargs)


class Instance(BaseMod):
    """Wrap an already instantiate module in an op factory

    It is used so that one

    factory = OpFactory(Instance(nn.Linear(2, 3)))
    same = factory.produce(<size>) is factory.produce(<size>)
    -> same is True
    """

    def __init__(self, module: ModInstance):
        """initializer

        Args:
            module (ModInstance): An instance of a module
        """

        self._module = module

    def to(self, **kwargs):
        module = self._module.to(**kwargs) if isinstance(self._module, arg) else self._module
        return module

    def produce(self, in_: typing.List[torch.Size], **kwargs):
        
        return self._module.to(**kwargs) if isinstance(self._module, arg) else self._module
    
    @property
    def module(self):
        return self._module

    def op(self, meta: Meta=None) -> OpFactory:
        """Convert to OpFactory

        Args:
            meta (Meta, optional): Meta information. Defaults to None.

        Returns:
            OpFactory: _description_
        """
        return OpFactory(self, meta)

    def to_ops(self) -> typing.List[OpFactory]:
        """Convert to list of NetFactories
        Returns:
            typing.List[OpFactory]
        """
        return [self.op()]


@singledispatch
def factory(
    mod: ModType, *args, _name: str='', 
    _meta: Meta=None, _out: typing.List[typing.List]=None, **kwargs
) -> OpFactory:
    """Convenience function for creating a factory

    Args:
        mod (ModType): The class / factory for the module
        _name (str, optional): Name of the factory. Defaults to ''.
        _meta (Meta, optional): Meta info for the factory. Defaults to None.
        _out (typing.List[typing.List], optional): Out size of the factory. Defaults to None.

    Returns:
        OpFactory
    """
    return OpFactory(ModFactory(mod, *args, *kwargs), name=_name, meta=_meta, _out=_out)


@factory.register
def _(mod: str, *args, _name: str='', _meta: Meta=None, _out: typing.List[typing.List]=None, **kwargs) -> OpFactory:
    """Convenience function for creating a factory where the module type is undefined

    Args:
        mod (str): Name of module factory
        _name (str, optional): Name of the factory. Defaults to ''.
        _meta (Meta, optional): Meta info for the factory. Defaults to None.
        _out (typing.List[typing.List], optional): Out size of the factory. Defaults to None.

    Returns:
        OpFactory
    """
    return OpFactory(ModFactory(argv(mod), *args, *kwargs), name=_name, meta=_meta, _out=_out)


@singledispatch
def instance(
    mod: ModInstance, _name: str='', _meta: Meta=None, 
    _out: typing.List[typing.List]=None
) -> OpFactory:
    """Convenience function for creating an instance factory

    Args:
        mod (ModInstance): The instance to 'produce'
        _name (str, optional): Name of the factory. Defaults to ''.
        _meta (Meta, optional): Meta info for the factory. Defaults to None.
        _out (typing.List[typing.List], optional): Out size of the factory. Defaults to None.

    Returns:
        OpFactory
    """
    return OpFactory(Instance(mod), name=_name, meta=_meta, _out=_out)


@instance.register
def _(
    mod: str, _name: str='', _meta: Meta=None, 
    _out: typing.List[typing.List]=None
) -> OpFactory:
    """Convenience function for creating an instance factory

    Args:
        mod (str): The naem of the instance to 'produce'. Will create an 'arg'
        _name (str, optional): Name of the factory. Defaults to ''.
        _meta (Meta, optional): Meta info for the factory. Defaults to None.
        _out (typing.List[typing.List], optional): Out size of the factory. Defaults to None.

    Returns:
        OpFactory
    """
    return OpFactory(Instance(argv(mod)), name=_name, meta=_meta, _out=_out)


class DivergeFactory(NetFactory):
    """Produces a DivergeModule
    """

    def __init__(
        self, op_factories: typing.List[NetFactory], name: str='', meta: Meta=None
    ):
        """initializer

        Args:
            op_factories (typing.List[NetFactory]): List of factories to 'diverge'
            name (str, optional): Name of the factory. Defaults to ''.
            meta (Meta, optional): Meta info for the factory. Defaults to None.
        """
        super().__init__(name, meta)
        self._op_factories = op_factories
    
    def to_ops(self) -> typing.List[OpFactory]:
        """Convert to list of NetFactories
        Returns:
            typing.List[OpFactory]
        """
        return [self]

    def produce(self, in_: typing.List[Out], **kwargs) -> typing.Tuple[Diverge, torch.Size]:
        """Produce a Diverge module

        Args:
            in_ (typing.List[Out]): The input size

        Returns:
            typing.Tuple[Diverge, torch.Size]: The Diverge module and the output size
        """
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
    def produce_nodes(self, in_: Multitap, namer: Namer=None, **kwargs) -> typing.Iterator[Node]:
        """Generator to produce nodes for diverging

        Args:
            in_ (Multitap): The input ports
            namer (Namer, optional): Defaults to None.

        Returns:
            typing.Iterator[Node]

        Yields:
            Iterator[typing.Iterator[Node]]: Each node the generator produces
        """
        namer = namer or FixedNamer()
        if isinstance(in_, Port):
            in_ = [in_]
        
        for in_i, op_factory in zip(in_, self._op_factories):
            for node in op_factory.produce_nodes(Multitap([in_i]), namer, **kwargs):
                yield node
    
    def to(self, **kwargs):
        """Update the args to the factory

        Returns:
            DivergeFactory: The factory with updated args
        """
        return DivergeFactory(
            [op_factory.to(**kwargs) for op_factory in self._op_factories],
            self._name, self._meta
        )

    def info_(
        self, name: str=None, labels: typing.List[str]=None, 
        annotation: str=None, *args, **kwargs
    ):
        """Update the 'info' for the DivergeFactory

        Args:
            name (str, optional): Name of the factory. Defaults to None.
            labels (typing.List[str], optional): Lables to the factory. Defaults to None.
            annotation (str, optional): Factory annotation. Defaults to None.

        Returns:
            DivergeFactory: Diverge factory with updated info
        """
        return DivergeFactory(self._op_factories, name or self._name, self._meta.spawn(labels, annotation))

    def to_ops(self):
        return [self]

diverge = DivergeFactory


class MultiFactory(NetFactory):
    """Factory to create a 'Multi' module
    """

    def __init__(
        self, op_factories: typing.List[NetFactory], name: str='', meta: Meta=None
    ):
        """initializer

        Args:
            op_factories (typing.List[NetFactory]): _description_
            name (str, optional): _description_. Defaults to ''.
            meta (Meta, optional): _description_. Defaults to None.
        """
        super().__init__(name, meta)
        self._op_factories = op_factories
    
    def to_ops(self):
        return [self]

    def produce(self, in_: typing.List[Out], **kwargs) -> typing.Tuple[Multi, torch.Size]:
        """Produce a Multi module

        Args:
            in_ (typing.List[Out]): The input size

        Returns:
            typing.Tuple[Multi, torch.Size]: The Multi module and the output size
        """
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
    def produce_nodes(self, in_: Multitap, namer: Namer=None, **kwargs) -> typing.Iterator[Node]:
        """Generator to produce nodes for MultiFactory

        Args:
            in_ (Multitap): The input ports
            namer (Namer, optional): Defaults to None.

        Returns:
            typing.Iterator[Node]

        Yields:
            Iterator[typing.Iterator[Node]]: Each node the generator produces
        """
        namer = namer or FixedNamer()
        for op_factory in self._op_factories:
            for node in op_factory.produce_nodes(in_, namer, **kwargs):
                yield node

    def to(self, **kwargs):
        """Convert 'args' in the MultiFactory

        Returns:
            MultiFactory
        """
        return MultiFactory(
            [op_factory.to(**kwargs) for op_factory in self._op_factories],
            self._name, self._meta
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        """ Update the 'info' for the MultiFactory

        Args:
            name (str, optional): Name of the factory. Defaults to None.
            labels (typing.List[str], optional): Lables to the factory. Defaults to None.
            annotation (str, optional): Factory annotation. Defaults to None.

        Returns:
            MultiFactory: Diverge factory with updated info
        """
        return MultiFactory(self._op_factories, name or self._name, self._meta.spawn(labels, annotation, fix))


multi = MultiFactory


class ChainFactory(NetFactory):
    def __init__(
        self, op_factory: NetFactory, attributes: typing.Union[arg, typing.List[dict], typing.List[Kwargs]],
        name: str='', meta: Meta=None
    ):
        super().__init__(name, meta)

        self._op_factory = op_factory
        self._attributes = attributes
    
    def to_ops(self):
        return [self]

    def produce(self, in_: typing.List[Size], **kwargs) -> typing.Tuple[nn.Sequential, torch.Size]:
        """Produce a Sequence of modules based on the factory and attributes

        Args:
            in_ (typing.List[Out]): The input size

        Returns:
            typing.Tuple[nn.Sequential, torch.Size]: The Chain module and the output size
        """
        if isinstance(in_, Size):
            in_ = [in_]
        attributes = self._attributes
        if isinstance(attributes, arg):
            attributes = self._attributes.to(**kwargs)

        mods = []
        out = in_
        for attribute in self._attributes:
            undefined = attribute.undefined
            if len(undefined) > 0:
                raise RuntimeError(f"Chain attributes {undefined} are not defined in {kwargs}")
        
            mod, out = self._op_factory.produce(out, **attribute.items)
            mods.append(mod)
        return nn.Sequential(*mods), out
    
    @to_multitap
    def produce_nodes(self, in_: Multitap, namer: Namer=None, **kwargs) -> typing.Iterator[Node]:
        """Generator to produce chain of nodes

        Args:
            in_ (Multitap): The input ports
            namer (Namer, optional): Defaults to None.

        Returns:
            typing.Iterator[Node]

        Yields:
            Iterator[typing.Iterator[Node]]: Each node the generator produces
        """
        namer = namer or FixedNamer()
        attributes = self._attributes
        if isinstance(attributes, arg):
            attributes = self._attributes.to(**kwargs)
        
        for attribute in self._attributes:
            if isinstance(attribute, dict):
                attribute = Kwargs(**attribute)
            
            attribute = attribute.lookup(in_, kwargs)

            undefined = attribute.undefined
            if len(undefined) > 0:
                raise RuntimeError(f"Chain attributes {undefined} are not defined in {kwargs}")

            for node in self._op_factory.produce_nodes(in_, namer, **attribute.items):
                yield node
                in_ = Multitap(node.ports)

    def to(self, **kwargs):
        """Update the args to the factory

        Returns:
            ChainFactory: The factory with updated args
        """
        attributes = self._attributes
        if isinstance(self._attributes, arg):
            attributes = self._attributes.to(**kwargs)
        
        to_attributes = []
        for attribute in attributes:
            to_attributes.append(attribute.remap_keys(kwargs))
            
        return ChainFactory(
            self._op_factory.to(**kwargs), to_attributes,
            self._name, 
            self._meta
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None):
        return ChainFactory(self._op_factory, self._attributes, name or self._name, self._meta.spawn(labels, annotation))

chain = ChainFactory


class InFactory(ABC):
    """Factory that produces In nodes
    """

    def __init__(self, name: str, meta: Meta=None):
        """initializer

        Args:
            name (str): Name of the factory
            meta (Meta, optional): Meta info for the factory. Defaults to None.
        """
        self._name = name
        self._meta = meta or Meta()

    @abstractmethod
    def produce(self, **kwargs) -> In:
        """Produce In node

        Returns:
            In: In node
        """
        raise NotImplementedError

    @abstractmethod
    def info_(
        self, name: str=None, labels: typing.List[str]=None, 
        annotation: str=None, *args, **kwargs
    ):
        pass


# class SizeVal(object):

#     def __init__(self, val: int, unknown: bool):

#         self._val = val
#         self._unkown = unknown

#     @property
#     def default(self):
#         return self._val

#     @property
#     def actual(self):
#         if self._unkown:
#             return -1

#         return self._val


def to_size(a: list):
    return [*map(lambda x: max(-1, x), a)]


def to_default_size(a: list):
    return [*map(abs, a)]


class TensorFactory(object):
    """Factory to create a tensor
    """

    def __init__(self, f, size: typing.List[int], kwargs: Kwargs=None):
        """initializer

        Args:
            f: Tensor creation function
            size (typing.List[int]): Size of the tensor
            kwargs (Kwargs, optional): Arguments to the tensor. Defaults to None.
        """
        self._f = f
        self._size = torch.Size(to_size(size))
        self._kwargs = kwargs or Kwargs()
        self._default_size = to_default_size(size)
        d = self.produce_default()
        self._dtype = d.dtype
        self._device = d.device
    
    @property
    def dtype(self) -> torch.dtype:
        """
        Returns:
            torch.dtype: dtype of the tenso
        """
        return self._dtype

    @property
    def device(self):
        """
        Returns:
            device of the tensor
        """
        return self._device

    def produce_default(self) -> torch.Tensor:
        return self._f(*self._default_size, **self._kwargs.items)

    @property
    def size(self) -> torch.Size:
        """
        Returns:
            torch.Size: size of the tensor
        """
        return torch.Size(self._size)
    
    def __call__(self, size=None) -> torch.tensor:
        """Produce the tensor

        Args:
            size (_type_, optional): Override size of the tensor. 
             Use default tensor if size=None

        Returns:
            torch.tensor: _description_
        """
        size = size or self._default_size
        return self._f(*size, **self._kwargs.items)


class TensorDefFactory(InFactory):
    """

    """

    def __init__(
        self, *size, dtype=torch.float, device='cpu', 
        default=None, name="", meta=None
    ):
        super().__init__(name or str("TensorIn"), meta)
        self._size = to_size(size)
        self._dtype = dtype
        self._device = device
        self._meta = meta or Meta()
        self._default = default or [1 if s < 0 else s for s in size]

        if default is not None:
            check_default = torch.zeros(self._size, device=self._device, dtype=self._dtype)
            self._check_size(check_default)

    def _check_size(self, default) -> bool:
        if len(default.size()) != len(self._size):
            raise ValueError(f'Size of default {default.size()} does not match size {self._size}')
        for s1, s2 in zip(default.size(), self._size):
            if s2 > 1 and s1 != s2:
                raise ValueError(f'Size of default {default.size()} does not match size {self._size}')

    def produce(self, namer: Namer=None) -> In:
        """Produces the In Node

        Args:
            namer (Namer, optional): Namer to name the tensor. Defaults to None.

        Returns:
            In: _description_
        """
        namer = namer or FixedNamer()
        if self._default:
            default = torch.tensor(self._default).to(self._device)
        else: default = None
        
        name = namer.name(self._name, default='TensorIn')
        return InTensor(
            name, torch.Size(self._size), self._dtype, default,
            meta=self._meta.spawn(), device=self._device
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        return TensorDefFactory(*self._size, dtype=self._dtype, device=self._device, name=name or self._name, meta=self._meta.spawn(labels, annotation, fix))


class TensorInFactory(InFactory):
    """Produces a TensorIn node from a function
    """

    def __init__(
        self, t: TensorFactory, name: str="", meta: Meta=None
    ):
        """initializer

        Args:
            t (TensorFactory): Produces a tensor
            name (str, optional): Name of the factory. Defaults to "".
            meta (Meta, optional): Meta info for the factory. Defaults to None.
        """
        super().__init__(name or "TensorIn", meta)
        self._t = t

    def produce(self, namer: Namer=None) -> In:
        """Produces the In Node

        Args:
            namer (Namer, optional): Namer to name the tensor. Defaults to None.

        Returns:
            In: _description_
        """
        namer = namer or FixedNamer()

        default = self._t.produce_default()
        size = self._t.size

        name = namer.name(self._name, default='TensorIn')
        return InTensor(
            name, size, self._t.dtype, default, meta=self._meta, device=self._t.device
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        return TensorInFactory(self._t, name or self._name, self._meta.spawn(labels, annotation))


class ScalarInFactory(InFactory):
    """
    """

    def __init__(self, type_: typing.Type, default: Any, call_default: bool=False, name: str="", meta: Meta=None):
        """initailzier

        Args:
            type_ (typing.Type): Scalar type
            default (Any): default Value
            call_default (bool, optional): Whether to call the default . Defaults to False.
            name (str, optional): Name of the factory. Defaults to "".
            meta (Meta, optional): Meta info for the factory. Defaults to None.
        """
        super().__init__(name or 'Scalar', meta)
        self._type_ = type_
        self._default = default
        self._call_default = call_default

    def produce(self, namer: Namer=None) -> In:
        namer = namer or FixedNamer()

        default = self._default() if self._call_default else self._default
        name = namer.name(name=self._name, default=type(self).__name__)
        return InScalar(name, default, self._meta.spawn())

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        
        return ScalarInFactory(
            self._type_, self._default, 
            self._call_default, 
            name or self._name,
            self._meta.spawn(labels, annotation, fix)
        )


def scalar_val(val, _meta: Meta=None):

    return ScalarInFactory(type(val), val, False, _meta)


def scalarf(f, type_: type, _meta: Meta=None):
    """[summary]

    Args:
        f ([type]): [description]
    """
    return ScalarInFactory(type_, f, True, _meta)


class ParameterFactory(InFactory):

    def __init__(self, t: TensorFactory, name: str="", meta: Meta=None):
        super().__init__(name or 'Param', meta)
        self._t = t

    def produce(self, namer: Namer=None) -> Node:

        namer = namer or FixedNamer()
        name = namer.name(self._name, default='ParamIn')

        return InTensor(
            name, self._t.size, self._t.dtype, 
            self._t.produce_default(), self._meta.spawn()
        )

    def info_(self, name: str=None, labels: typing.List[str]=None, annotation: str=None, fix: bool=None):
        return ParameterFactory(
            self._t, 
            name or self._name,
            self._meta.spawn(labels, annotation, fix)
        )


class TakoMod(ABC):
    """Convenience for creating an op factory like a normal module

    opnn = OpMod(nn)
    opnn.Linear(1, 4) <- This will output an OpFactory with a linear
    """

    @abstractproperty
    def __call__(self, *args, **kwargs) -> NetFactory:
        raise NotImplementedError


class NNMod(TakoMod):
    """Create OpFactories for an nn.Module
    """

    def __init__(self, nnmodule: typing.Type[nn.Module]):

        self._nnmodule = nnmodule

    def __call__(self, *args, _meta: Meta=None, **kwargs) -> OpFactory:
        
        return OpFactory(ModFactory(self._nnmodule, *args, **kwargs), _meta)


class TensorMod(TakoMod):
    
    def __init__(self, tensor_mod):
        """
        Args:
            tensor_mod (typing.Callable): mod for creating a tensor
        """

        self._tensor_mod = tensor_mod

    def __call__(self, *size, _meta: Meta=None, **kwargs) -> TensorInFactory:
        """
        Returns:
            TensorInFactory: Tensor factory based on size and meta info
        """
        try:
            factory = TensorFactory(
                self._tensor_mod, size, Kwargs(**kwargs)
            )
        except Exception as e:
            raise RuntimeError(f'Could not generate tensor with {self._tensor_mod}.') from e

        return TensorInFactory(factory,  _meta)


class ParamMod(TakoMod):
    """Create Parameter factory using torch functions
    """
    
    def __init__(self, parameter_mod):
        """
        Args:
            tensor_mod (typing.Callable): mod for creating a parameter
        """
        self._parameter_mod = parameter_mod

    def __call__(self, *size, _meta: Meta=None, **kwargs) -> ParameterFactory:
        try: 
            factory = TensorFactory(self._parameter_mod, size, Kwargs(**kwargs))
        except Exception as e:
            raise RuntimeError(
                f'Could not generate tensor with {self._parameter_mod}.'
            ) from e

        return ParameterFactory(factory, _meta)


class OpMod(object):
    """Convenience for creating an op factory like a normal module

    opnn = OpMod(nn)
    opnn.Linear(1, 4) <- This will output an OpFactory with a linear

    optorch = OpMod(torch, TensorMod)
    optorch.zeros(1, 2, dtype=torch.float) <- Create a TensorInFactory
    """

    def __init__(self, mod, factory: TakoMod=None):
        self._mod = mod
        self._factory = factory or NNMod

    def __getattribute__(self, __name: str) -> TakoMod:
        mod = super().__getattribute__('_mod')
        factory = super().__getattribute__('_factory')
        name = getattr(mod, __name)

        try:
            return factory(name)
        except Exception as e:
            raise RuntimeError(f'Could not get {factory} {name} for {mod}') from e


def port_to_multitap(vals):
    """Convert port or set of ports to a multitap

    Raises:
        ValueError: If type is not valid

    Returns:
        Multitap
    """
    ports = []

    if isinstance(vals, Port):
        return Multitap([vals])

    for v in vals:
        if isinstance(v, str):
            ports.append(NodePort(v))
        elif isinstance(v, Port):
            ports.append(v)
        elif isinstance(v, Multitap):
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
    def __init__(self, namer: Namer=None):
        self._net = Network()
        self._names = Counter()
        self._namer = namer if namer is not None else CounterNamer()

    def __getitem__(self, keys):
        return self._net[keys].ports

    def add_in(self, *in_args: InFactory, **in_kwargs: InFactory):
        ports = []
        for in_i in in_args:
            node = in_i.produce(self._namer)
            self._net.add_node(node)
            ports.extend(node.ports)
        for k, in_k in in_kwargs.items():
            node = in_k.info_(k).produce(self._namer)
            self._net.add_node(node)
            ports.extend(node.ports)
        return ports

    def append(self, to, factory: NetFactory, **kwargs):
        multitap = port_to_multitap(to)
        ports = []
        for node in factory.produce_nodes(multitap, self._namer, **kwargs):
            ports = self._net.add_node(node)
        return ports
    
    def interface(self, out, by: typing.List[str]):
        return NetworkInterface(self._net, out, by)

    @property
    def net(self):
        return self._net

    @property
    def namer(self):
        return self._namer

    def set_default_interface(
        self, ins: typing.List[typing.Union[Port, str]], outs: typing.List[typing.Union[Port, str]]
    ):
        self._net.set_default_interface(
            ins, outs
        )

    # TODO: Figure out how to implement
    # add in port mapping
    # def build_machine(self, name: str, *learner_mixins: typing.Type[MachineComponent], **name_map):

    #     class _(*learner_mixins):
    #         __qualname__ = name

    #     return _(self.net)
