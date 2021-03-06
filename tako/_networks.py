"""
Network and its components to build up a graph.

A network is a graph of nodes which process the input
"""
from abc import ABC, abstractmethod, abstractproperty
from inspect import Parameter
from re import T
import torch.nn as nn
import torch
import typing
import copy
import dataclasses
import itertools
from functools import singledispatch, singledispatchmethod
from dataclasses import dataclass, field


class LabelSet(object):
    """Labels to tag a module
    """
    
    def __init__(self, labels: typing.Iterable[str]=None):
        """initializer

        Args:
            labels (typing.Iterable[str], optional): Labels to tag a node with. Defaults to None.
        """
        labels = labels or []
        self._labels = set(labels)
    
    def __contains__(self, key):
        return key in self._labels
    
    @property
    def labels(self):
        return [*self._labels]
    
    def add(self, label: str):
        self._labels.add(label)


@dataclass
class Meta:
    """Supplementary info for a module. Can be extended to add more info
    """

    labels: LabelSet=field(default_factory=LabelSet)
    annotation: str=''

    def __post_init__(self):
        if isinstance(self.labels, typing.List):
            self.labels = LabelSet(self.labels)
    
    def spawn(self, labels: typing.List[str]=None, annotation: str=None, **kwargs):
        return Meta(
            LabelSet(labels) if labels is not None else self.labels,
            annotation if annotation is not None else self.annotation,
        )


class By(object):
    """Stores the outputs of nodes in the network and 
    the network's subnets.
    """

    def __init__(self, **data):
        """initializer

        Args:
          data: Initialize the ouputs of nodes 
        """
        self._data = data
        self._subs = {}

    # TODO: add in port index
    def get(self, node: str, default=None) -> typing.Union[torch.Tensor, typing.List[torch.Tensor]]:
        """get output of a node

        Args:
            node (str): Node to get the output for
            default (_type_, optional): Default value to return. Defaults to None.

        Returns:
            typing.Union[torch.Tensor, typing.List[torch.Tensor]]: _description_
        """
        
        return self._data.get(node, default)

    # TODO: add in port index
    def update(self, node: str, datum):
        """Update output of a node

        Args:
            node (str): node to update
            datum (_type_): value to update with
        """
        self._data[node] = datum

    def __contains__(self, node: str) -> bool:
        """Check if node's output is defined
        """
        return node in self._data

    def __setitem__(self, node: str, datum):
        """Update the output of a node
        """
        self._data[node] = datum

    def __getitem__(self, node: str):
        """Get the output of a node
        """
        return self._data[node]

    def get_or_add_sub(self, sub: str):
        """Get or add a sub network
        """
        if sub not in self._subs:
            self._subs[sub] = By()
        return self._subs[sub]


def check_size(x_size: typing.Iterable[int], port_size: typing.Iterable[int]) -> typing.Optional[str]:
    """Check if the input size matches the size of the ports

    Args:
        x_size (typing.Iterable[int]): The input size
        port_size (typing.Iterable[int]): The size of the port

    Returns:
        typing.Optional[str]: Error string if the sizes do not match
    """
    if len(x_size) != len(port_size):
        return f"Dimensions of input size {x_size} does not match port size {port_size}"
    
    for i, (x_i, p_i) in enumerate(zip(x_size, port_size)):
        if p_i == -1:
            continue
        if x_i != p_i:
            return f"Size of input {x_size} at dim {i} must match size of ports {port_size}"
    return None


@dataclasses.dataclass
class Out:
    """Definition of the output of a node
    """

    size: torch.Size
    dtype: torch.dtype=torch.float

    def __post_init__(self):
        if not isinstance(self.size, torch.Size):
            self.size = torch.Size(self.size)


OutList = typing.List[Out]


class Port(ABC):
    """A port into or out of a node. Used for connecting nodes together."""

    @abstractproperty
    def node(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def size(self) -> torch.Size:
        raise NotImplementedError

    @abstractproperty
    def dtype(self) -> torch.dtype:
        raise NotImplementedError

    @abstractmethod
    def select(self, by: By, check: bool=False):
        raise NotImplementedError

    @abstractmethod
    def select_result(self, result):
        raise NotImplementedError


class NodePort(Port):
    """Port for a node with a single port
    """

    def __init__(self, node: str, size: torch.Size, dtype: torch.dtype=torch.float):
        """initializer

        Args:
            node (str): Name of node
            size (torch.Size): _description_
            dtype (torch.dtype, optional): _description_. Defaults to torch.float.
        """

        self._node = node
        self._size = size
        self._dtype = dtype

    @property
    def node(self) -> str:
        """
        Returns:
            str: Name of the node for the port
        """
        return self._node

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns:
            torch.dtype: Dtype of the port
        """
        return self._dtype

    @property
    def size(self) -> str:
        """
        Returns:
            torch.Size: Size of the port
        """
        return self._size

    def select(self, by: By, check: bool=False) -> torch.Tensor:
        """
        Args:
            by (By)

        Returns:
            torch.Tensor: Output of thenode
        """
        x = by.get(self.node)

        if x is None or not check:
            return x

        check_res = check_size(x, self.size)
        if check_res is None:
            return x
        raise ValueError(f'For mod {self._node}, {check_res}')

    def select_result(self, result) -> torch.Tensor:
        """Select result of an output

        Args:
            result (torch.Tensor): Output of a ndoe

        Returns:
            torch.Tensor
        """
        return result


class IndexPort(Port):

    def __init__(self, node: str, index: int, size: torch.Size, dtype: torch.dtype=torch.float):
        """initializer

        Args:
            node (str): Name of node
            index (int): Index of the port
            size (torch.Size): _description_
            dtype (torch.dtype, optional): _description_. Defaults to torch.float.
        """
        self._node = node
        self._index = index
        self._size = size
        self._dtype = dtype

    @property
    def node(self) -> str:

        """
        Returns:
            str: Name of the node for the port
        """
        return self._node

    @property
    def index(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self._index

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns:
            torch.dtype: Dtype of the port
        """
        return self._dtype

    @property
    def size(self) -> str:
        """
        Returns:
            torch.Size: Size of the port
        """
        return self._size

    def select(self, by: By, check: bool=False):
        """
        Args:
            by (By)

        Returns:
            torch.Tensor: Output of the port for the index
        """
        result = by.get(self.node)
        # TODO:
        # by.get(self.node, self.index)
        # TODO:
        # For "Get" nodes.. i need some way to check
        # if i can the output without executing the node
        # i think i do need to use probe rather than
        # forward
        if result is None:
            return None

        x = result[self.index]
        if not check:
            return x

        check_res = check_size(x, self.size)
        if check_res is None:
            return x
        raise ValueError(f'For mod {self._node} - index {self._index}, {check_res}')

    def select_result(self, result) -> torch.Tensor:
        """Select result of an output at the index

        Args:
            result (typing.List[torch.Tensor]): Output of a node

        Returns:
            torch.Tensor
        """
        return result[self.index]


@dataclasses.dataclass
class Multitap:
    """A set of ports
    """

    ports: typing.List[Port]

    def __getitem__(self, idx: int):
        return self.ports[idx]
    
    @singledispatchmethod
    def __getitem__(self, idx: typing.Iterable):
        return Multitap([self.ports[i] for i in idx])

    @__getitem__.register
    def _(self, idx: int) -> Port:
        return self.ports[idx]
    
    def __iter__(self) -> typing.Iterator[Port]:
        for port in self.ports:
            yield port

    @property
    def sizes(self):
        return [port.size for port in self.ports]

    def select(self, by: typing.Dict):
        result = []
        for port in self.ports:
            cur = port.select(by)
            if isinstance(cur, list):
                result.extend(cur)
            else:
                result.append(cur)
        return result
    
    def clone(self):
        return Multitap([*self.ports])

    @classmethod
    def from_nodes(cls, nodes: typing.List):
        result = []
        for node in nodes:
            result.extend(node.ports)
        return cls(result)


class Node(nn.Module):
    """Base class for a node in the network
    """

    def __init__(
        self, name: str, meta: Meta=None
    ):
        """_summary_

        Args:
            name (str): _description_
            meta (Meta, optional): _description_. Defaults to None.
        """
        super().__init__()
        self._name = name
        self._meta = meta or Meta()

    @property
    def name(self) -> str:
        return self._name
    
    def add_label(self, label: str):
        self._meta.labels.add(label)
    
    @property
    def labels(self) -> typing.List[str]:
        return copy.copy(self._meta.labels)
    
    @property
    def annotation(self) -> str:
        return self._meta.annotation

    @annotation.setter
    def annotation(self, annotation: str) -> str:
        self._meta.annotation = annotation

    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        raise NotImplementedError

    @property
    def ports(self) -> typing.Iterable[Port]:
        raise NotImplementedError
    
    @property
    def inputs(self) -> Multitap:
        raise NotImplementedError
    
    def clone(self):
        raise NotImplementedError

    # TODO: I think i need to remove since I'll probably not use
    def accept(self, visitor):
        raise NotImplementedError

    def probe(self, by: By, to_cache: True):
        """Probe the output of the node

        Args:
            by (By): 
            to_cache (True): Whether to store the outputs
        """
        raise NotImplementedError


class NodeSet(object):
    """Set of nodes in the network. Makes it more convenient to 
    retrieve ports in the network

    Usage:

    # get the nodeset
    nodeset = network[['x', 'y']]
    
    # output the ports
    nodeset.ports

    Args:
        object (_type_): _description_
    """

    def __init__(self, nodes: typing.List[Node]):

        self._order = [node.name for node in nodes]
        self._nodes: typing.Dict[str, Node] = {node.name: node for node in nodes}
        self.__or__ = self.unify
        self.__and__ = self.intersect
        self.__sub__ = self.difference

    @property
    def nodes(self) -> typing.List[Node]:
        """
        Returns:
            typing.List[Node]: Nodes in the nodeset
        """

        return [
            self._nodes[name]
            for name in self._order
        ]

    @property
    def ports(self):
        """
        Returns:
            Multitap: Set of ports for the node set
        """

        return Multitap.from_nodes(self.nodes)

    @singledispatchmethod
    def __getitem__(self, key: typing.Union[str, int, list]):
        raise TypeError("Key passed in must be of type string, int or list")

    @__getitem__.register
    def _(self, key: str):
        return self._nodes[key]
    
    @__getitem__.register
    def _(self, key: list):
        return NodeSet(
            [self[k] for k in key]
        )

    @__getitem__.register
    def _(self, key: int):
        return self._nodes[key]
    
    def unify(self, other):
        """Unify two node sets

        Args:
            other (NodeSet)

        Returns:
            _type_: NodeSet
        """

        nodes = [node for _, node in self._nodes.items()]

        for name in other._order:
            if name not in self._nodes:
                nodes.append(other._nodes[name])
        
        return NodeSet(nodes)
    
    def intersect(self, other):
        
        """Intersect two node sets

        Args:
            other (NodeSet)

        Returns:
            _type_: NodeSet
        """
        nodes = []
        for name in self._order:
            if name in other._nodes:
                nodes.append(self._nodes[name])
        
        return NodeSet(nodes)

    def difference(self, other):
        """Get difference between two node sets

        Args:
            other (NodeSet)

        Returns:
            _type_: NodeSet
        """
        nodes = []
        for name in self._order:
            if name not in other._nodes:
                nodes.append(self._nodes[name])
        return NodeSet(nodes)

# TODO: Add OpNode subclass for get

class OpNode(Node):
    """
    A node that performs an operation.
    """
    def __init__(
        self, name: str, operation: nn.Module, 
        inputs: typing.Union[Multitap, Port, typing.List[Port]],
        outs: typing.Union[Out, OutList], meta: Meta=None
    ):
        super().__init__(name, meta)
        if isinstance(inputs, Port):
            inputs = Multitap([inputs])
        elif isinstance(inputs, list):
            inputs = Multitap(inputs)
        
        self._outs = outs
        self._inputs: Multitap = inputs
        self.op: nn.Module = operation
    
    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return [in_.node for in_ in self._inputs]
    
    @property
    def ports(self) -> typing.Iterable[Port]:
        """
        example: 
        # For a node with two ports
        x, y = node.ports
        # You may use one port as the input to one node and the
        # other port for another node

        Returns:
            typing.Iterable[Port]: [The output ports for the node]
        """
        if isinstance(self._outs, list):
            return [IndexPort(self.name, i, out.size, out.dtype) for i, out in enumerate(self._outs)]

        return NodePort(self.name, self._outs.size, self._outs.dtype),

    @property
    def inputs(self) -> Multitap:
       return self._inputs.clone()

    def clone(self):
        return OpNode(
            self.name, self.op, self._inputs, self._outs, self._info.spawn()
        )
    
    def forward(self, *args, by: By=None):

        result = self.op(*args)
        if by is not None:
            by[self.name] = result
        return result

    # TODO: Add in ports to probe.. order them correctly
    def probe(self, by: By, to_cache=True):
        """probe the node

        Args:
            by (By): The excitations in the network
            to_cache (bool, optional): Whether to cache the result or not. Defaults to True.

        Raises:
            RuntimeError: Could not probe the network

        Returns:
            _type_: _description_
        """
        if self.name in by:
            return [in_.select(by) for in_ in self.inputs]
            # return by[self.name]

        try:
            result = self.op(*[in_.select(by) for in_ in self.inputs])
        except Exception as e:
            raise RuntimeError(f'Could not probe node {self.name} {type(self.op)}') from e
        if to_cache:
            by[self.name] = result
        return result


class InDef(nn.Module):

    def __init__(self, dtype, default, size: torch.Size=None):
        super().__init__()
        self.default = default
        self.dtype = dtype
        self.size = size or torch.Size([])

    def redefine(self, value):
        return InDef(type(value), value, torch.Size([]))


class TensorDef(InDef):

    def __init__(
        self, *size, dtype: torch.dtype=torch.float, 
        default: typing.Union[torch.Tensor, nn.parameter.Parameter] = None
    ):
        super().__init__(dtype, default, size)
        if isinstance(default, torch.Tensor):
            self.default = nn.parameter.Parameter(self.default)

    def redefine(self, value: torch.Tensor):
        return TensorDef(*value.shape, dtype=value.dtype, default=value)


class Undef(InDef):
    
    def __init__(self):
        super().__init__(None, None, None)
    
    def redefine(self, value):
        if isinstance(value, torch.Tensor):
            return TensorDef(*value.shape, dtype=value.dtype, default=value)
        return InDef(type(value), value, torch.Size([]))
        


class UndefinedSize(object):
    """Used in order to specify a parameter in the network that is undefined.
    Using an InNode with this and a Setter node will update the Setter to use 
    this value on creation
    """

    def __init__(self, in_):

        self._in: In = in_

    def define(self, value):

        self._in.define(value)


class In(Node):

    def __init__(
        self, name: str, def_: InDef, meta: Meta=None
    ):
        """[initializer]

        Args:
            name ([type]): [Name of the in node]
            out_size (torch.Size): [The size of the in node]
        """
        super().__init__(name, meta)

        self._def = def_
        self._default = def_.default

    def forward(self, x: torch.Tensor):
        return x

    @property
    def ports(self) -> typing.Iterable[Port]:
        if self._def is Undef:
            return NodePort(self.name, UndefinedSize(self), None)
        return NodePort(self.name, self._def.size, self._def.dtype),

    def clone(self):
        return In(
            self.name, self._def, self._meta.spawn()
        )
    
    def redefine(self, value):
        self._def = self._def.redefine(value)

    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return []

    @property
    def inputs(self) -> Multitap:
        return Multitap([])

    @property
    def default(self):
        """Default value for the input
        """
        return self._default
    
    @default.setter
    def default(self, value: torch.Tensor):
        """Default value for the input
        """
        self._default = value

    def probe(self, by: By, to_cache: bool=True):
        return by.get(self.name, self.default)


class Network(nn.Module):
    """
    Network of nodes. Use for building complex machines.
    """

    # Set teh namer
    def __init__(self, inputs: typing.List[In]=None):
        super().__init__()

        self._leaves: typing.Set[str] = set()
        self._roots: typing.Set[str] = set()
        self._nodes: nn.ModuleDict = nn.ModuleDict()
        self._node_outputs: typing.Dict[str, typing.List[str]] = {}
        
        for in_ in inputs or []:
            self.add_node(in_)

    # def add_in() <- uses a factory

    def add_node(self, node: Node):
        if node.name in self._nodes:
            raise ValueError("Node {} already exists in the network.".format(node.name))
        
        for input_node in node.input_nodes:
            if input_node not in self._nodes:
                raise ValueError(
                    "There is no node named for input {} in the network.".format(input_node))
            
            self._node_outputs[input_node].append(node.name)
        
        if len(node.input_nodes) == 0:
            self._roots.add(node.name)
        
        for input_node in node.input_nodes:
            if input_node in self._leaves:
                self._leaves.remove(input_node)
        self._leaves.add(node.name)

        self._nodes[node.name] = node
        self._node_outputs[node.name] = []
        return node.ports

    # def append() <- uses a factory

    @property
    def leaves(self) -> typing.Iterator[Node]:
        
        for name in self._leaves:
            yield self._nodes[name]

    @property
    def roots(self) -> typing.Iterator[Node]:
        
        for name in self._roots:
            yield self._nodes[name]
    
    # def op
    
    # TODO: Remove
    def get_node(self, key: str) -> Node:

        return self._nodes[key]

    def traverse_forward(
        self, from_nodes: typing.List[str]=None, 
        to_nodes: typing.Set[str]=None
    ) ->  typing.Iterator[Node]:

        if from_nodes is None:
            from_nodes = self._roots
        
        for node_name in from_nodes:
            node: Node = self._nodes[node_name]
            yield node

            if to_nodes is None or node_name not in to_nodes:
                self.traverse_forward(self._node_outputs[node], to_nodes)

    def traverse_backward(
        self, from_nodes: typing.List[str]=None, 
        to_nodes: typing.Set[str]=None
    ) -> typing.Iterator[Node]:
        
        if from_nodes is None:
            from_nodes = self._default_outs
        
        for node_name in from_nodes:
            node: Node = self._nodes[node_name]
            yield node

            if to_nodes is None or node_name not in to_nodes:
                self.traverse_backward(node.inputs, to_nodes)
    
    def _probe_helper(
        self, node: Node, by: By, to_cache=True, check_size: bool=False
    ):
        """Helper function to get the output for a "probe"

        Args:
            node (Node): Node to probe
            excitations (typing.Dict[str, torch.Tensor]): Current set of excitations

        Raises:
            KeyError: A node does not exist

        Returns:
            torch.Tensor: Output of a node
        """
        if node.name in by:
            return by[node.name]

        for port in node.inputs:
            try:
                if port.select(by, check_size) is None:
                    # need to get the result
                    # TODO: as it is right now. it requires caching
                    # result = self._probe_helper() <- include by
                    # results.append(result) or results.extend(result)
                    self._probe_helper(self._nodes[port.node], by)
                    # port.select(by, check_size)
                
                # TODO: Once I retrieve the input then
                # check the port to see if the size is correct


            # TODO: Create a better report for this
            except KeyError:
                raise KeyError(f'Input or Node {port.node} does not exist')

        # TODO: not using "inputs".. should use forward here, I think
        # cur_result = node.forward(*inputs, by, to_cache, check_size)
        #  ... i could turn caching off in 'by' also.. wouldn't
        # that be better?
        # forward function will cache the results if to cache is true
        cur_result = node.probe(by, to_cache)
        # pass in the cache to store 
        # cure_result = node.forward(*inputs, by=by)
        return cur_result

    def probe(
        self, outputs: typing.List[typing.Union[str, Port]], 
        by: typing.Dict[str, torch.Tensor], to_cache: bool=True, check_size: bool=False
    ) -> typing.List[torch.Tensor]:
        """Probe the network for its inputs

        Args:
            outputs (typing.List[str, Port]): The nodes or Port to probe
            by (typing.Dict[str, torch.Tensor]): The values to input into the network

        Returns:
            typing.List[torch.Tensor]: The outptus for the probe
        """
        if isinstance(by, dict):
            by = By(**by)
        
        if not isinstance(outputs, list):
            singular = True
            outputs = [outputs]
        else: singular = False

        result = []

        for output in outputs:
            if isinstance(output, Port):
                cur = self._probe_helper(
                    self._nodes[output.node], 
                    by, to_cache, check_size)
                result.append(
                    output.select_result(cur)
                )
            else:
                result.append(
                    self._probe_helper(
                        self._nodes[output], 
                        by, to_cache
                ))
        
        if singular:
            return result[0]
        return result

    @singledispatchmethod
    def __getitem__(self, name):
        return self._nodes[name]
    
    @__getitem__.register
    def _(self, name: list):
        return NodeSet([self._nodes[val] for val in name])

    # TODO: Look how to simplify
    @__getitem__.register
    def _(self, name: tuple):
        return NodeSet([self._nodes[val] for val in name])

    @__getitem__.register
    def _(self, name: str):
        return self._nodes[name]

    def __contains__(self, name: str) -> bool:
        return name in self._nodes

    def __iter__(self) -> typing.Tuple[str, Node]:
        """Iterate over all nodes

        Returns:
            Node: a node in the network
        """
        for k, v in self._nodes.items():
            yield k, v

Network.forward = Network.probe


@dataclasses.dataclass
class Link:

    from_: Port
    to_: Port

    def map(self, from_dict: typing.Dict, to_dict: typing.Dict):
        map_val = self.from_.select(from_dict)
        to_dict[self.to_.node] = map_val


class SubNetwork(object):
    """A network wrapper which can be added to another network and used as a node through InterfaceNode"""

    def __init__(
        self, name: str, network: Network, 
        meta: Meta=None
    ):
        super().__init__()
        self._name = name
        self._network: Network = network
        self._info = meta or Meta()
    
    @property
    def ports(self) -> typing.Iterable[Port]:
        return []

    @property
    def inputs(self) -> Multitap:
        return Multitap([])
    
    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return []
    
    def __getitem__(self, name) -> NodeSet:
        return self._network[name]
    
    def clone(self):
        return SubNetwork(
            self._name, self._network, self._info.spawn()
        )
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def probe(
        self, outputs: typing.List[str], 
        inputs: typing.List[Link], 
        by: By, to_cache=True
    ):
        if isinstance(by, dict):
            by = By(**by)
        sub_by = by.get_or_add_sub(self._name)
        
        for link in inputs:
            link.map(by, sub_by)
        
        probe_results = self._network.probe(outputs, sub_by, to_cache)

        if to_cache:
            by[self._name] = probe_results

        return probe_results


class InterfaceNode(Node):
    """Node that is an interface to a network
    """

    def __init__(
        self, name: str, sub_network: SubNetwork, 
        outputs: Multitap,
        inputs: typing.List[Link],
        meta: Meta=None
    ):
        """initializer

        Args:
            name (str): Name of the interface
            sub_network (SubNetwork): Network to interface to
            outputs (Multitap): Ports used for the output
            inputs (typing.List[Link]): Inputs into the network
            meta (Meta, optional): Defaults to None.
        """
        super().__init__(name, meta)
        self._sub_network: SubNetwork = sub_network
        self._outputs: Multitap = outputs
        self._inputs: typing.List[Link] = inputs

    @property
    def ports(self) -> typing.Iterable[Port]:
        """
        example: 
        # For a node with two ports
        x, y = node.ports
        # You may use one port as the input to one node and the
        # other port for another node

        Returns:
            typing.Iterable[Port]: [The output ports for the node]
        """
        return [IndexPort(self.name, i, port.size) for i, port in enumerate(self._outputs)]

    @property
    def inputs(self) -> Multitap:
        return Multitap([in_.from_ for in_ in self._inputs])
    
    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return [in_.from_.node for in_ in self._inputs]
    
    @property
    def sub_network(self) -> SubNetwork:
        return self._sub_network
    
    @property
    def outputs(self) -> Multitap:
        return self._outputs.clone()
    
    @property
    def clone(self):
        return InterfaceNode(
            self._name, self._sub_network,
            self._outputs, self._inputs, self._info.spawn()
        )

    def probe(self, by: By, to_cache=True) -> typing.List[torch.Tensor]:
        """Probe the sub network

        Args:
            by (typing.Dict): 
            to_cache (bool, optional): Whether to cache the output. Defaults to True.

        Returns:
            list[Tensor]
        """

        if self.name in by:
            return by[self.name]
        
        output_names = [output.node for output in self._outputs]
        subnet_result = dict(zip(output_names, self._sub_network.probe(
            output_names, self._inputs, by, to_cache
        )))

        result = []
        for output in self._outputs:
            result.append(output.select(subnet_result))

        if to_cache:
            by[self.name] = result
        
        return result

    
class NetworkInterface(nn.Module):
    """A module for probing a network with a standard interface"""

    def __init__(
        self, network: Network, out: list, by: typing.List[str]
    ):
        super().__init__()
        self._network = network
        if isinstance(by, str):
            self._by = [by]
        self._by = by
        self._out = out

    def forward(self, *x):
        by = dict(zip(self._by, x))
        return self._network.probe(self._out, by=by)

    def __add__(self, other):

        return NetworkInterface(
            self._network, self._out + other._out, self._by + other._by
        )
