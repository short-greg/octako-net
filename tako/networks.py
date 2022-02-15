
# TODO: Rewrite tests and get everything working

"""
Classes related to Networks.

They are a collection of modules connected together in a graph.

"""
from abc import ABC, abstractmethod, abstractproperty
import torch.nn as nn
import torch
import typing
import copy
import dataclasses
import itertools
from functools import singledispatch, singledispatchmethod
from dataclasses import dataclass, field


class LabelSet:
    
    def __init__(self, labels: typing.Iterable[str]=None):
        labels = labels or []
        self._labels = set(labels)
    
    def __contains__(self, key):
        return key in self._labels
    
    @property
    def labels(self):
        return [*self._labels]


@dataclass
class Meta:

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

    def __init__(self, **kwargs):
        self._data = kwargs
        self._subs = {}

    def get(self, node: str, default=None):
        return self._data.get(node, default)

    def update(self, node: str, datum):
        self._data[node] = datum

    def __contains__(self, node: str):
        return node in self._data

    def __setitem__(self, node: str, datum):
        self._data[node] = datum

    def __getitem__(self, node: str):
        return self._data[node]

    def get_or_add_sub(self, sub: str):
        if sub not in self._subs:
            self._subs[sub] = By()
        return self._subs[sub]


@dataclasses.dataclass
class Out:

    size: torch.Size
    dtype: torch.dtype=torch.float


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
    def select(self, by: By):
        raise NotImplementedError

    @abstractmethod
    def select_result(self, result):
        raise NotImplementedError


class NodePort(Port):

    def __init__(self, node: str, size: torch.Size, dtype: torch.dtype=torch.float):

        self._node = node
        self._size = size
        self._dtype = dtype

    @property
    def node(self) -> str:
        return self._node

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def size(self) -> str:
        return self._size

    def select(self, by: By):
        return by.get(self.node)

    def select_result(self, result):
        return result


class IndexPort(Port):

    def __init__(self, node: str, index: int, size: torch.Size, dtype: torch.dtype=torch.float):

        self._node = node
        self._index = index
        self._size = size
        self._dtype = dtype

    @property
    def node(self) -> str:
        return self._node

    @property
    def index(self) -> str:
        return self._index

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def size(self) -> str:
        return self._size

    def select(self, by: By):
        result = by.get(self.node)
        if result is None:
            return None

        return result[self.index]

    def select_result(self, result):
        return result[self.index]


@dataclasses.dataclass
class Multitap:

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

    def __init__(
        self, name: str, meta: Meta=None
    ):
        super().__init__()
        self._name = name
        self._info = meta or Meta()

    @property
    def name(self):
        return self._name
    
    def add_label(self, label: str):
        self._info.labels.append(label)
    
    @property
    def labels(self) -> typing.List[str]:
        return copy.copy(self._info.labels)
    
    @property
    def annotation(self) -> str:
        return self._info.annotation

    @annotation.setter
    def annotation(self, annotation: str) -> str:
        self._info.annotation = annotation

    def cache_names_used(self):
        raise NotImplementedError

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

    def accept(self, visitor):
        raise NotImplementedError

    def probe(self, by: typing.Dict, to_cache: True):
        raise NotImplementedError


class NodeSet(object):

    def __init__(self, nodes: typing.List[Node]):

        self._order = [node.name for node in nodes]
        self._nodes: typing.Dict[str, Node] = {node.name: node for node in nodes}
        self.__or__ = self.unify
        self.__and__ = self.intersect
        self.__sub__ = self.difference

    @property
    def nodes(self) -> typing.List[Node]:

        return [
            self._nodes[name]
            for name in self._order
        ]

    @property
    def ports(self):
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
        nodes = [node for _, node in self._nodes.items()]

        for name in other._order:
            if name not in self._nodes:
                nodes.append(other._nodes[name])
        
        return NodeSet(nodes)
    
    def intersect(self, other):
        
        nodes = []
        for name in self._order:
            if name in other._nodes:
                nodes.append(self._nodes[name])
        
        return NodeSet(nodes)

    def difference(self, other):
        
        nodes = []
        for name in self._order:
            if name not in other._nodes:
                nodes.append(self._nodes[name])
        return NodeSet(nodes)


class NodeVisitor(object):

    @singledispatch
    def visit(self, node: Node):
        pass


class OpNode(Node):
    """
    A node in a network. It performs an operation and specifies which 
    nodes connect into it.
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

    # TODO: FIND OUT WHY NOT WORKING
    @property
    def inputs(self) -> Multitap:
       return self._inputs.clone()
    
    @property
    def cache_names_used(self) -> typing.Set[str]:
        return set([self.name])

    def clone(self):
        return OpNode(
            self.name, self.op, self._inputs, self._outs, self._info.spawn()
        )
    
    def forward(self, *args, **kwargs):

        return self.op(*args, **kwargs)

    def probe(self, by: By, to_cache=True):
        
        # need to check if all inputs in by
        if self.name in by:
            return by[self.name]

        try:
            result = self.op(*[in_.select(by) for in_ in self.inputs])
        except Exception as e:
            raise RuntimeError(f'Could not probe node {self.name} {type(self.op)}') from e
        if to_cache:
            by[self.name] = result
        return result


class In(Node):
    """[Input node in a network.]"""

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
    
    @property
    def default(self):
        raise NotImplementedError
    
    @property
    def cache_names_used(self) -> typing.Set[str]:
        return set([self.name])
    
    def probe(self, by: By, to_cache: bool=True):
        return by.get(self.name, self.default)


class InTensor(In):

    def __init__(
        self, name: str, sz: torch.Size, 
        dtype: typing.Union[type, torch.dtype], 
        default: torch.Tensor=None, 
        meta: Meta=None,
        device: str='cpu'
    ):
        """[initializer]

        Args:
            name ([type]): [Name of the in node]
            out_size (torch.Size): [The size of the in node]
        """
        super().__init__(name, meta)
        self._dtype = dtype
        self._out_size = sz
        self._device = device
        self._default = default
        
        if self._default is not None: 
            self._default = self._default.to(device)
        self._device = device

    def forward(x: torch.Tensor):
        return x

    def to(self, device):
        self._device = device
        if self._default is not None:
            self._default = self._default.to(device)

    @property
    def ports(self) -> typing.Iterable[Port]:
        return NodePort(self.name, self._out_size, self._dtype),

    def clone(self):
        return InTensor(
            self.name, self._out_size, self._dtype, self._default, self._info.spawn()
        )

    @property
    def default(self) -> torch.Tensor:
        return self._default
    

class InScalar(In):

    def __init__(
        self, name, dtype: typing.Type, default, meta: Meta=None
    ):
        super().__init__(name, meta)
        self._dtype = dtype
        self._default = default

    def forward(x):
        return x

    def to(self, device):
        pass

    @property
    def ports(self) -> typing.Iterable[Port]:
        return NodePort(self.name, torch.Size([]), self._dtype),

    @classmethod
    def from_scalar(
        cls, name, default_type: typing.Type, default_value, 
        labels: LabelSet=None, annotation: str=None):

        return cls(
            name, torch.Size([]), default_type, 
            default_value, labels, annotation
        )


class Network(nn.Module):
    """
    Network of nodes. Use for building complex machines.
    """

    def __init__(self, inputs: typing.List[In]=None):
        super().__init__()

        self._leaves: typing.Set[str] = set()
        self._roots: typing.Set[str] = set()
        self._nodes: nn.ModuleDict = nn.ModuleDict()
        self._node_outputs: typing.Dict[str, typing.List[str]] = {}
        self._default_ins: typing.List[str] = []
        self._default_outs: typing.List[str] = []
        self._cache_names_used: typing.Set[str] = set()
        for in_ in inputs or []:
            self.add_node(in_)

    @property
    def output_names(self):
        return copy.copy(self._default_outs)

    @property
    def input_names(self):
        return copy.copy(self._default_ins)
    
    def set_default_interface(self, ins: typing.List[typing.Union[Port, str]], outs: typing.List[typing.Union[Port, str]]):
        
        self._default_ins = []
        self._default_outs = []

        for in_ in ins:
            if isinstance(in_, str):
                self._default_ins.append(in_)
            else:
                self._default_ins.append(in_.node)
        
        for out in outs:
            if isinstance(out, str):
                self._default_outs.append(out)
            else:
                self._default_outs.append(out.node)

    def add_node(self, node: Node):
        if node.name in self._nodes:
            raise ValueError("Node {} already exists in the network.".format(node.name))
        
        for input_node in node.input_nodes:
            if input_node not in self._nodes:
                raise ValueError(
                    "There is no node named for input {} in the network.".format(input_node))
            
            self._node_outputs[input_node].append(node.name)
        
        if len(self._cache_names_used.intersection(node.cache_names_used)) > 0:
            raise ValueError("Cannot add node {} because the cache names are already used.".format(node.name))
        
        if len(node.input_nodes) == 0:
            self._roots.add(node.name)
        
        for input_node in node.input_nodes:
            if input_node in self._leaves:
                self._leaves.remove(input_node)
        self._leaves.add(node.name)

        print(list(self._nodes.keys()))
        self._nodes[node.name] = node
        self._node_outputs[node.name] = []
        return node.ports
    
    @property
    def nodes(self) -> typing.Iterator[Node]:

        for node in self._nodes:
            yield node

    @property
    def leaves(self) -> typing.Iterator[Node]:
        
        for name in self._leaves:
            yield self._nodes[name]

    @property
    def roots(self) -> typing.Iterator[Node]:
        
        for name in self._roots:
            yield self._nodes[name]
    
    def _get_input_names_helper(self, node: Node, use_input: typing.List[bool], roots: typing.List):

        for node_input_port in node.inputs:
            name = node_input_port.node
            try:
                use_input[roots.index(name)] = True
            except ValueError:
                self._get_input_names_helper(self._nodes[name], use_input, roots)

    def get_input_names(self, output_names: typing.List[str]) -> typing.List[str]:
        """
        Args:
            output_names (typing.List[str]): Output names in the network.
        Returns:
            typing.List[str]: The names of all the inputs required for the arguments.
        """
        use_input = [False] * len(self._roots)
        assert len(use_input) == len(self._roots)

        for output_name in output_names:
            if output_name not in self._nodes:
                raise KeyError(f'Output name {output_name} is not in the network')

        roots = list(self._roots)
        for name in output_names:
            self._get_input_names_helper(self._nodes[name], use_input, roots)

        return [input_name for input_name, to_use in zip(self._roots, use_input) if to_use is True]

    def _is_input_name_helper(
        self, node: Node, input_names: typing.List[str], 
        is_inputs: typing.List[bool]
    ) -> bool:
        """Helper to check if the node is an input for an output

        Args:
            node (Node): A node in the network
            input_names (typing.List[str]): Current input names
            is_inputs (typing.List[bool]): A list of booleans that specifies
            which nodes are inputs
        Returns:
            bool: Whether a node is an input
        """
        other_found = False
        if not node.inputs:
            return True

        for node_input_port in node.inputs:
            name = node_input_port.node

            try:
                is_inputs[input_names.index(name)] = True
            except ValueError:
                other_found = self._is_input_name_helper(self._nodes[name], input_names, is_inputs)
                if other_found: break
        
        return other_found

    def are_inputs(self, output_names: typing.List[str], input_names: typing.List[str]) -> bool:
        """Check if a list of nodes are directly or indirectly inputs into other nodes

        Args:
            output_names (typing.List[str]): Names of nodes to check
            input_names (typing.List[str]): Names of input candidates

        Raises:
            KeyError: Name of the module

        Returns:
            bool: Whether or not input_names are inputs
        """
        
        is_inputs = [False] * len(input_names)

        for name in itertools.chain(input_names, output_names):
            if name not in self._nodes:
                raise KeyError(f'Node name {name} does not exist')
    
        for name in output_names:
            other_found: bool = self._is_input_name_helper(self._nodes[name], input_names, is_inputs)

            if other_found:
                break
        all_true = not (False in is_inputs)
        return all_true and not other_found
    
    # TODO: Update traverse forward / traverse backward
    # to just return an iterator over nodes
    def traverse_forward(self, visitor: NodeVisitor, from_nodes: typing.List[str]=None, to_nodes: typing.Set[str]=None):

        if from_nodes is None:
            from_nodes = self._roots
        
        for node_name in from_nodes:
            node: Node = self._nodes[node_name]
            node.accept(visitor)

            if to_nodes is None or node_name not in to_nodes:
                self.traverse_forward(visitor, self._node_outputs[node], to_nodes)

        # TODO: consider whether to include the subnetwork

    def traverse_backward(self, visitor: NodeVisitor, from_nodes: typing.List[str]=None, to_nodes: typing.Set[str]=None):
        
        if from_nodes is None:
            from_nodes = self._default_outs
        
        for node_name in from_nodes:
            node: Node = self._nodes[node_name]
            node.accept(visitor)

            if to_nodes is None or node_name not in to_nodes:
                self.traverse_backward(visitor, node.inputs, to_nodes)

        # TODO: Add in subnetwork
    
    def _probe_helper(
        self, node: Node, by: By, to_cache=True
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

        inputs = []

        for port in node.inputs:
            node_name = port.node
            excitation = port.select(by)

            if excitation is not None:
                
                inputs.append(excitation)
                continue
            try:
                self._probe_helper(self._nodes[node_name], by)
                inputs.append(
                    port.select(by)
                )
            # TODO: Create a better report for this
            except KeyError:
                raise KeyError(f'Input or Node {node_name} does not exist')

        cur_result = node.probe(by, to_cache)
        return cur_result

    def probe(
        self, outputs: typing.List[typing.Union[str, Port]], 
        by: typing.Dict[str, torch.Tensor], to_cache=True
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
                    by, to_cache)
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

    def forward(self, *args, **kwargs) -> typing.List[torch.Tensor]:
        """The standard 'forward' method for the network.

        Returns:
            typing.List[torch.Tensor]: Outputs of the network
        """

        # TODO: UPDATE THIS FORWARD FUNCTION 
        if (len(args) != len(self._default_ins)):
            raise ValueError(f"Number of args {len(args)} does not match the number of inputs {len(self._default_ins)}'")
        inputs = {self._default_ins[i]: x for i, x in enumerate(args)}
        inputs.update(kwargs)
        return self.probe(self._default_outs, inputs)


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

    def accept(self, visitor: NodeVisitor):
        visitor.visit(self)
    
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

    def __init__(
        self, name: str, sub_network: SubNetwork, 
        outputs: Multitap,
        inputs: typing.List[Link],
        meta: Meta=None
    ):
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
    def cache_names_used(self) -> typing.Set[str]:
        return set([self.name, self._sub_network.name])

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

    def accept(self, visitor: NodeVisitor):
        visitor.visit(self)

    def probe(self, by: typing.Dict, to_cache=True):

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
    """A node for probing a network with a standard interface"""

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

# query - port 1, node
# meta - {1: torch.Tensor} 
# need to almagamate all queries for a node into one selector


# StepBy <- inherit from by
