from collections import namedtuple
import torch.nn as nn
import torch
import typing
import copy
import dataclasses
import itertools
from abc import ABC, abstractmethod
from functools import singledispatch, singledispatchmethod

from octako.machinery import builders
from octako.machinery.utils import coalesce
from octako.modules.utils import Lambda


# TODO: Rewrite tests and get everything working

"""
Classes related to Networks.

They are a collection of modules connected together in a graph.

"""


@dataclasses.dataclass
class ModRef(object):
    module: str

    def select(self, by: typing.Dict):
        
        return by.get(self.module)


@dataclasses.dataclass
class IndexRef(ModRef):
    index: int
    def select(self, by: typing.Dict):
        result = super().select(by)
        if result is None:
            return None

        return result[self.index]


@dataclasses.dataclass
class Port:
    """A port into or out of a node. Used for connecting nodes together.
    """
    
    # TODO: Decide whether to add a name to the port
    # name: str
    ref: ModRef
    size: torch.Size

    @property
    def module(self):
        return self.ref.module

    def select(self, by: typing.Dict):

        return self.ref.select(by)
    
    @staticmethod
    def merge_results(ports, by: typing.Dict):
        result = []
        for port in ports:
            cur = port.select(by)
            if isinstance(cur, list):
                result.extend(cur)
            else:
                result.append(cur)
        return result


@dataclasses.dataclass
class NetworkPort(Port):
    """A port into or out of a node. Used for connecting nodes together.
    """
    
    network: str

    @property
    def module(self):
        return self.network

    def select(self, by: typing.Dict):

        sub_by = by.get(self.network)
        if self.network is None:
            return None

        return self.ref.select(sub_by)


@dataclasses.dataclass
class Operation:
    """An operation performed and its output size. Used in creating the network.
    """

    op: nn.Module
    out_size: torch.Size


class Node(nn.Module):

    def __init__(
        self, name: str, 
        labels: typing.List[typing.Union[typing.Iterable[str], str]]=None,
        annotation: str=None
    ):
        super().__init__()
        self.name: str = name
        self._labels = labels
        self._annotation = annotation or ''
    
    def add_label(self, label: str):
        self._labels.append(label)
    
    @property
    def labels(self) -> typing.List[str]:
        return copy.copy(self._labels)
    
    @property
    def annotation(self) -> str:
        return self._annotation

    @annotation.setter
    def annotation(self, annotation: str) -> str:
        self._annotation = annotation

    @property
    def ports(self) -> typing.Iterable[Port]:
        raise NotImplementedError
    
    @property
    def cache_names_used(self):
        raise NotImplementedError

    @property
    def inputs(self) -> typing.List[Port]:
        raise NotImplementedError
    
    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        raise NotImplementedError
    
    def clone(self):
        raise NotImplementedError

    def accept(self, visitor):
        raise NotImplementedError

    def probe(self, by: typing.Dict, to_cache: True):
        raise NotImplementedError


class NodeSet(object):

    def __init__(self, nodes: typing.List[Node]):

        self._nodes = nodes
        self._node_dict: typing.Dict[str, Node] = {node.name: node for node in nodes}

    @property
    def ports(self):

        result = []
        for node in self._nodes:
            result.extend(node.ports)
        return result

    @singledispatchmethod
    def __getitem__(self, key: typing.Union[str, int, list]):
        raise TypeError("Key passed in must be of type string, int or list")

    @__getitem__.register
    def _(self, key: str):
        return self._node_dict[key]
    
    @__getitem__.register
    def _(self, key: list):
        return NodeSet(
            [self[k] for k in key]
        )

    @__getitem__.register
    def _(self, key: int):
        return self._nodes[key]


class NodeVisitor(object):

    @singledispatch
    def visit(self, node: Node):
        pass


class OpNode(Node):
    """
    A node in a network. It performs an operation and specifies which 
    modules connect into it.
    """

    def __init__(
        self, name: str, operation: nn.Module, 
        inputs: typing.List[Port],
        out_size: typing.Union[torch.Size, typing.List[torch.Size]],
        labels: typing.List[typing.Union[typing.Iterable[str], str]]=None,
        annotation: str=None
    ):
        super().__init__(name, labels, annotation)
        self.operation: nn.Module = operation
        self._out_size = out_size
        self._inputs: typing.List[Port] = inputs

    @property
    def ports(self) -> typing.Iterable[Port]:
        """
        example: 
        # For a node with two ports
        x, y = node.ports
        # You may use one port as the input to one module and the
        # other port for another module

        Returns:
            typing.Iterable[Port]: [The output ports for the node]
        """

        if type(self._out_size) == list:
            return [Port(IndexRef(self.name, i), sz) for i, sz in enumerate(self._out_size)]

        return Port(ModRef(self.name), self._out_size),
    
    @property
    def inputs(self) -> typing.List[Port]:
        return self._inputs
    
    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return [in_.module for in_ in self.inputs]
    
    @property
    def cache_names_used(self) -> typing.Set[str]:
        return set([self.name])

    def clone(self):
        return OpNode(
            self.name, self.operation, self._inputs, self._out_size, self._labels,
            self._annotation
        )
    
    def forward(self, *args, **kwargs):

        return self.operation(*args, **kwargs)

    def probe(self, by: typing.Dict, to_cache=True):
        
        if self.name in by:
            return by[self.name]

        result = self.operation(*[in_.select(by) for in_ in self._inputs])
        if to_cache:
            by[self.name] = result
        return result


class In(Node):
    """[Input node in a network.]"""

    def __init__(
        self, name, sz: torch.Size, value_type: typing.Type, default_value, labels: typing.List[typing.Union[typing.Iterable[str], str]]=None, annotation: str=None):
        """[initializer]

        Args:
            name ([type]): [Name of the in node]
            out_size (torch.Size): [The size of the in node]
        """
        super().__init__(name, labels=labels, annotation=annotation)
        self._value_type = value_type
        self._out_size = sz
        self._default_value = default_value

    def to(self, device):
        if self._value_type == torch.Tensor:
            self._default_value = self._default_value.to(device)

        self._default_value = self._default_value.to(device)

    @property
    def ports(self) -> typing.Iterable[Port]:

        return Port(ModRef(self.name), self._out_size),
    
    def forward(x):

        return x

    @property
    def inputs(self) -> typing.List[Port]:
        return []

    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return []

    def clone(self):
        return In(
            self.name, self._out_size, self._value_type, self._default_value, self._labels,
            self._annotation

        )
    
    @property
    def cache_names_used(self) -> typing.Set[str]:
        return set([self.name])
    
    def probe(self, by: typing.Dict, to_cache: bool=True):
        # TODO: Possibly check the value in by
        return by.get(self.name, self._default_value)

    @classmethod
    def from_tensor(cls, name, sz: torch.Size, default_value: torch.Tensor=None, labels: typing.List[typing.Union[typing.Iterable[str], str]]=None, annotation: str=None):
        if default_value is None:
            sz2 = []
            for el in list(sz):
                if el == -1:
                    sz2.append(1)
                else:
                    sz2.append(el)
            if len(sz2) != 0:
                default_value = torch.zeros(*sz2)
            else:
                default_value = torch.tensor([])
        return cls(name, sz, torch.Tensor, default_value, labels, annotation)
    
    @classmethod
    def from_scalar(cls, name, default_type: typing.Type, default_value, labels: typing.Set[str]=None, annotation: str=None):

        return cls(
            name, torch.Size([]), default_type, default_value, labels, annotation
        )


class Parameter(Node):
    """[Input node in a network.]"""

    def __init__(
        self, name: str, sz: torch.Size, reset_func: typing.Callable[[torch.Size], torch.Tensor], labels: typing.List[typing.Union[typing.Iterable[str], str]]=None, annotation: str=None):
        """[initializer]

        Args:
            name ([type]): [Name of the in node]
            out_size (torch.Size): [The size of the in node]
        """
        super().__init__(name, labels=labels, annotation=annotation)
        self._reset_func = reset_func
        self._out_size = sz
        self._value = self._reset_func(self._out_size)

    def reset(self):
        self._value = self._reset_func(self._out_size)

    def to(self, device):
        self._value = self._value.to(device)

    @property
    def ports(self) -> typing.Iterable[Port]:
        return Port(ModRef(self.name), self._out_size),
    
    def forward(x):

        return x

    @property
    def inputs(self) -> typing.List[Port]:
        return []

    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return []

    def clone(self):
        return Parameter(
            self.name, self._out_size, self._reset_func, self._labels,
            self._annotation

        )
    
    @property
    def cache_names_used(self) -> typing.Set[str]:
        return set([self.name])
    
    def probe(self, by: typing.Dict, to_cache: bool=True):
        return by.get(self.name, self._value)


class Network(nn.Module):
    """
    Network of nodes. Use for building complex machines.
    """

    def __init__(self, inputs: typing.List[In]=None):
        super().__init__()

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
                self._default_ins.append(in_.module)
        
        for out in outs:
            if isinstance(out, str):
                self._default_outs.append(out)
            else:
                self._default_outs.append(out.module)
    
    def is_name_taken(self, name):
        return name in self._nodes

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
        
        self._nodes[node.name] = node
        self._node_outputs[node.name] = []
        return node.ports

    def get_node(self, name: str, t: typing.Type=None):
        node = self._nodes.get(name)
        if t is None or isinstance(node, t):
            return node
    
    def get_ports(self, names: typing.Union[str, typing.List[str]], flat=True) -> typing.List[Port]:

        if isinstance(names, str):
            return self._nodes[names].ports
        
        ports = []
        for name in names:
            if name not in self._nodes:
                raise ValueError(f'There is no node named {name} in the network')
            if flat:
                ports.extend(self._nodes[name].ports)
            else:
                ports.append(self._nodes[name].ports)

        return ports
    
    def nodes_by_label(self, labels: typing.Iterable[str]):

        for node in self._nodes:
            node: Node = node
            
            if len(set(node.labels).intersection(labels)) == len(labels):
                yield node
    
    def _get_input_names_helper(self, node: Node, use_input: typing.List[bool]):

        for node_input_port in node.inputs:
            name = node_input_port.module
            try:
                use_input[self._in_names.index(name)] = True
            except ValueError:
                self._get_input_names_helper(self._nodes[name], use_input)

    def get_input_names(self, output_names: typing.List[str]) -> typing.List[str]:
        """
        Args:
            output_names (typing.List[str]): Output names in the network.
        Returns:
            typing.List[str]: The names of all the inputs required for the arguments.
        """
        use_input = [False] * len(self._in_names)
        assert len(use_input) == len(self._in_names)

        for output_name in output_names:
            if output_name not in self._nodes:
                raise KeyError(f'Output name {output_name} is not in the network')

        for name in output_names:
            self._get_input_names_helper(self._nodes[name], use_input)

        return [input_name for input_name, to_use in zip(self._in_names, use_input) if to_use is True]

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
            name = node_input_port.module

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
    
    def traverse_forward(self, visitor: NodeVisitor, from_nodes: typing.List[str]=None, to_nodes: typing.Set[str]=None):

        if from_nodes is None:
            from_nodes = self._in_names
        
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
        self, node: Node, by: typing.Dict[str, torch.Tensor], to_cache=True
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
            module_name = port.module
            excitation = port.select(by)

            if excitation is not None:
                inputs.append(excitation)
                continue
            try:
                self._probe_helper(self._nodes[module_name], by)
                inputs.append(
                    port.select(by)
                )
            # TODO: Create a better report for this
            except KeyError:
                raise KeyError(f'Input or Node {module_name} does not exist')

        cur_result = node.probe(by, to_cache)
        # excitations[node.name] = cur_result
        return cur_result

    def probe(
        self, outputs: typing.List[str], by: typing.Dict[str, torch.Tensor], to_cache=True
    ) -> typing.List[torch.Tensor]:
        """Probe the network for its inputs

        Args:
            outputs (typing.List[str]): The nodes to probe
            by (typing.Dict[str, torch.Tensor]): The values to input into the network

        Returns:
            typing.List[torch.Tensor]: The outptus for the probe
        """

        if isinstance(outputs, str):
            outputs = [outputs]

        excitations = {**by}
        result = {}

        for output in outputs:
            node = self._nodes[output]
            cur_result = self._probe_helper(node, excitations, to_cache)
            result[output] = cur_result
        
        return result
    
    # TODO: Depracate
    def get_node(self, key) -> Node:
        return self._nodes[key]

    @singledispatchmethod
    def __getitem__(self, name):
        return self._nodes[name]
    
    @__getitem__.register
    def _(self, name: list):
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
        result_dict = self.probe(self._default_outs, inputs)
        result = [result_dict[out] for out in self._default_outs]
        return result


@dataclasses.dataclass
class Link:

    from_: Port
    to_: Port

    def map(self, from_dict: typing.Dict, to_dict: typing.Dict):
        map_val = self.from_.select(from_dict)
        to_dict[self.to_.module] = map_val


class SubNetwork(object):
    """
    """

    def __init__(
        self, name: str, network: Network, 
        labels: typing.List[typing.Union[typing.Iterable[str], str]]=None,
        annotation: str=None
    ):
        super().__init__()
        self._network: Network = network
        self._name = name
        self._labels = labels
        self._annotation = annotation
    
    def get_ports(self, node_name: str) -> typing.List[Port]:

        return self._network.get_ports(node_name)

    @property
    def ports(self) -> typing.Iterable[Port]:
        return []

    @property
    def inputs(self) -> typing.List[Port]:
        return []
    
    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return []
    
    def clone(self):
        return SubNetwork(
            self._name, self._network, 
            self._labels, self._annotation
        )

    def accept(self, visitor: NodeVisitor):
        visitor.visit(self)
    
    @property
    def name(self, name: str):
        self._name = name

    def probe(
        self, outputs: typing.List[str], inputs: typing.List[Link], 
        by: typing.Dict, to_cache=True
    ):
        if self._name not in by:
            by[self._name] = {}
        
        sub_by = by[self._name]
        for link in inputs:
            link.map(by, sub_by)
        
        probe_results = self._network.probe(outputs, sub_by, to_cache)
        for output in outputs:
            sub_by[output] = probe_results[output]

        if to_cache:
            by[self._name] = probe_results

        return probe_results


class InterfaceNode(Node):

    def __init__(
        self, name: str, sub_network: SubNetwork, 
        outputs: typing.List[Port],
        inputs: typing.List[Link],
        labels: typing.List[typing.Union[typing.Iterable[str], str]]=None,
        annotation: str=None
    ):
        super().__init__(name, labels, annotation)
        self._sub_network: SubNetwork = sub_network
        self._outputs: typing.List[Port] = outputs
        self._inputs: typing.List[Link] = inputs

    @property
    def ports(self) -> typing.Iterable[Port]:
        """
        example: 
        # For a node with two ports
        x, y = node.ports
        # You may use one port as the input to one module and the
        # other port for another module

        Returns:
            typing.Iterable[Port]: [The output ports for the node]
        """
        return [Port(IndexRef(self.name, i), port.size) for i, port in enumerate(self._outputs)]

    @property
    def cache_names_used(self) -> typing.Set[str]:
        return set([self.name, self._sub_network.name])

    @property
    def inputs(self) -> typing.List[Port]:
        # use self._inputs
        return [in_.from_ for in_ in self._inputs]
    
    @property
    def input_nodes(self) -> typing.List[str]:
        """
        Returns:
            typing.List[str]: Names of the nodes input into the node
        """
        return [in_.from_.module for in_ in self._inputs]
    
    @property
    def sub_network(self) -> SubNetwork:
        return self._sub_network
    
    @property
    def outputs(self) -> typing.List[Port]:
        return [output for output in self._outputs]
    
    @property
    def clone(self):
        return InterfaceNode(
            self.name, self._sub_network,
            self._outputs, self._inputs, self._labels, self._annotation
        )

    def accept(self, visitor: NodeVisitor):
        visitor.visit(self)

    # TODO: should have a cache decorator
    def probe(self, by: typing.Dict, to_cache=True):
        if self.name in by:
            return by[self.name]
        
        output_names = [output.module for output in self._outputs]

        subnet_result = self._sub_network.probe(output_names, self._inputs, by, to_cache)
        result = []
        for output in self._outputs:
            result.append(output.select(subnet_result))

        if to_cache:
            by[self.name] = result
        
        return result

    
class NetworkInterface(nn.Module):

    def __init__(
        self, network: Network, inputs: typing.List[str], outputs: typing.List[str]
    ):
        super().__init__()
        self._network = network
        self._inputs = inputs
        self._outputs = outputs

    def forward(self, *x):
        
        by = dict(zip(self._inputs, x))
        return self._network.probe(self._outputs, by)
