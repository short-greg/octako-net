from collections import namedtuple
import torch.nn as nn
import torch
import typing
import copy
import dataclasses
import itertools
from abc import ABC, abstractmethod
from functools import singledispatch


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


class NodeVisitor(object):

    @singledispatch
    def visit(self, node):
        pass


class Node(nn.Module):

    def __init__(
        self, name: str, 
        labels: typing.List[str]=None,
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

    def accept(self, visitor: NodeVisitor):
        raise NotImplementedError

    def probe(self, by: typing.Dict, to_cache: True):
        raise NotImplementedError


class OperationNode(Node):
    """
    A node in a network. It performs an operation and specifies which 
    modules connect into it.
    """

    def __init__(
        self, name: str, operation: nn.Module, 
        inputs: typing.List[Port],
        out_size: typing.Union[torch.Size, typing.List[torch.Size]],
        labels: typing.List[str]=None,
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

    def clone(self):
        return OperationNode(
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

    def __init__(self, name, sz: torch.Size, value_type: typing.Type, default_value, labels: typing.Set[str]=None, annotation: str=None):
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
    
    def probe(self, by: typing.Dict):
        # TODO: Possibly check the value in by
        return by.get(self.name, self._default_value)


    @staticmethod
    def from_tensor(name, sz: torch.Size, default_value: torch.Tensor=None, labels: typing.Set[str]=None, annotation: str=None):
        if default_value is None:
            sz2 = []
            for el in list(sz):
                if el == -1:
                    sz2.append(1)
                else:
                    sz2.append(el)
            default_value = torch.zeros(*sz2)
        return In(name, sz, torch.Tensor, default_value, labels, annotation)
    
    @staticmethod
    def from_scalar(name, default_type: typing.Type, default_value, labels: typing.Set[str]=None, annotation: str=None):

        return In(
            name, torch.Size([]), default_type, default_value, labels, annotation
        )


class Network(nn.Module):
    """
    Network of nodes. Use for building complex machines.
    """

    def __init__(self, inputs: typing.List[In]=None):
        super().__init__()

        self._nodes: nn.ModuleDict = nn.ModuleDict()

        inputs = inputs or []
        self._in_names: typing.List[str] = []

        for in_ in inputs:
            self.add_input(in_)
    
        self._default_ins: typing.List[str] = [] # [in_.name for in_ in self._ins]
        self._default_outs: typing.List[str] = []
        self._networks = []
        self._node_outputs = {}

    @property
    def output_names(self):
        return copy.copy(self._default_outs)

    @property
    def input_names(self):
        return copy.copy(self._default_ins)
    
    def add_input(self, in_: In) -> typing.Iterable[Port]:
        if in_.name in self._nodes:
            raise ValueError(f'There is already a node with the name {in_.name} ')

        self._nodes[in_.name] = in_
        self._in_names.append(in_.name)
        return in_.ports
    
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
        return name in self._networks or name in self._nodes
    
    def _validate_node_or_network_name(self, name):
        if self.is_name_taken(name):
            raise KeyError(f'Network or node with {name} already exists.')

    def add_subnetwork(self, name, network, labels: typing.List[str]=None, annotation: str=None):
        self._validate_node_or_network_name(name)
        self._networks[name] = SubNetwork(name, network, labels, annotation)

    def add_network_interface(
        self, name: str, network_name: str, 
        outputs: typing.List[Port], inputs: typing.List,
        labels: typing.List[str]=None, annotation: str=None
    ):
        inputs: typing.List[Link] = inputs
        self._validate_node_or_network_name(name)
        interface = NetworkInterface(name, self._networks[network_name], outputs, inputs, labels, annotation)
        self._nodes[name] = interface.name

        input_ports = [in_.from_ for in_ in inputs]
        self._set_node_outputs(input_ports, name)

        return interface.ports

    def add_node(
        self, name: str, op: Operation, in_: typing.Union[Port, typing.List[Port]], 
        labels: typing.List[str]=None
    ) -> typing.List[Port]:
        """[summary]

        Args:
            name (str): The name of the node
            op (Operation): Operation for the node to perform
            in_ (typing.Union[Port, typing.List[Port]]): The ports feeding into the node
            labels (typing.List[str], optional): Labels for the node to be used for searching. Defaults to None. 

        Raises:
            KeyError: If the name for the node already exists in the network.

        Returns:
            typing.List[Port]: The ports feeding out of the node
        """

        self._validate_node_or_network_name(name)

        if type(in_) == Port:
            in_ = [in_]

        self._set_node_outputs(in_, name)

        # assert (is_input and not len(inputs) > 0) or (not is_input and len(inputs) > 0)
        node = OperationNode(name, op.op, in_, op.out_size, labels)
        self._nodes[name] = node
        return node.ports
    
    def _set_node_outputs(self, in_ports: typing.List[Port], name): 

        for port in in_ports:
            if port.module not in self._nodes:
                raise ValueError(
                    f"There is no node named for input {port.module} in the network.")
            
            self._node_outputs[port.module].append(name)
        
        self._node_outputs[name] = []
    
    def get_input_ports(self) -> typing.Iterable[Port]:
        ports = []

        for in_ in [self._nodes[in_name] for in_name in self._in_names]:
            
            ports.extend(in_.ports)
        return ports
    
    def get_ports(self, name) -> typing.List[Port]:

        if name not in self._nodes:
            raise KeyError(f'There is no node named {name} in the network')

        return self._nodes[name].ports
    
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

        # cur_result = node(*inputs)
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
        result  = []

        for output in outputs:
            node = self._nodes[output.module]
            cur_result = self._probe_helper(node, excitations)
            output.select()
            result[output] = cur_result
        
        return result
            
    def get_node(self, key) -> Node:

        return self._nodes[key]

    def __iter__(self) -> typing.Tuple[str, Node]:
        """Iterate over all nodes

        Returns:
            Node: a node in the network
        """
        for k, v in self._nodes.items():
            return k, v

    def forward(self, *args, **kwargs) -> typing.List[torch.Tensor]:
        """The standard 'forward' method for the network.

        Returns:
            typing.List[torch.Tensor]: Outputs of the network
        """

        # TODO: UPDATE THIS FORWARD FUNCTION 
        inputs = {self._default_ins[i]: x for i, x in enumerate(args)}
        inputs.update(kwargs)
        result_dict = self.probe(self._default_outs, inputs)
        result = [result_dict[out] for out in self._default_outs]
        return result


@dataclasses.dataclass
class Link:

    from_: typing.Union[Port, typing.List[Port]]
    to_: Port

    def __post_init__(self):

        if isinstance(self.from_, Port):
            self.map = self._map_single
        else:
            self.map = self._map_multiple

    def _map_multiple(self, from_dict: typing.Dict, to_dict: typing.Dict):
        to_dict[self.to_.module] = Port.merge_results(self.from_, from_dict)

    def _map_single(self, from_dict: typing.Dict, to_dict: typing.Dict):
        map_val = self.from_.select(from_dict)
        to_dict[self.to_.module] = map_val

    def map(self, from_dict: typing.Dict, to_dict: typing.Dict):
        # Placeholder method
        pass
    


class SubNetwork(object):
    """
    """

    def __init__(
        self, name: str, network: Network, 
        labels: typing.List[str]=None,
        annotation: str=None
    ):
        super().__init__()
        self._network: Network = network
        self._name = name
        self._labels = labels
        self._annotation = annotation
    
    def get_ports(self, node_name: str) -> typing.List[Port]:

        return self._network.get_ports(node_name)
    
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
        self, outputs: typing.List[Port], inputs: typing.List[Link], 
        by: typing.Dict, to_cache=True
    ):
        if self._name not in by:
            by[self._name] = {}
        
        sub_by = by[self._name]
        for link in inputs:
            link.map(by, sub_by)

        probe_results = self._network.probe(outputs, sub_by, to_cache)
        result = Port.merge_results(outputs, probe_results)
        by[self._name] = result
        return result


class NetworkInterface(OperationNode):

    def __init__(
        self, name: str, sub_network: SubNetwork, 
        outputs: typing.List[Port],
        inputs: typing.List[Link],
        labels: typing.List[str]=None,
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
    
    def clone(self):
        return NetworkInterface(
            self.name, self._sub_network,
            self._outputs, self._inputs, self._labels, self._annotation
        )

    def accept(self, visitor: NodeVisitor):
        visitor.visit(self)

    # TODO: should have a cache decorator
    def probe(self, by: typing.Dict, to_cache=True):
        if self.name in by:
            return by[self.name]

        result = self._sub_network.probe(self._outputs, self._inputs, by, to_cache)
        if to_cache:
            by[self.name] = result
        
        return result


class NodeProcessor(ABC):
    
    @abstractmethod
    def process_input_node(self, node: In) -> In:
        pass

    @abstractmethod
    def process_operation_node(self, node: OperationNode) -> OperationNode:
        pass


class UpdateNodeName(NodeProcessor):

    def __init__(self, prepend_with='', append_with=''):

        self._prepend_with = prepend_with
        self._append_with = append_with
        self.process_input_node = self._update_name
        self.process_operation_node = self._update_name

    def _update_name(self, node: Node):

        name = self._prepend_with + node.name + self._append_with
        node = node.clone()
        node.name = name
        return node


class NullNodeProcessor(NodeProcessor):

    def __init__(self):
        pass

    def process_input_node(self, node: In):
        return node.clone()
    
    def process_operation_node(self, node: Node):
        return node.clone()

class NetInterface(typing.NamedTuple):

    inputs: typing.List[typing.Union[Port, str]]
    outputs: typing.List[typing.Union[Port, str]]


class NetworkBuilder(object):

    def __init__(self, node_processor: NodeProcessor):
        self._network = None
    
        self._operation_nodes: typing.Dict[str, OperationNode] = None
        self._node_processor = node_processor
        # self._prepend_name = prepend_name
        self._added_nodes: typing.Set[str] = None
        self.reset()
    
    def reset(self):

        self._network = None
        self._operation_nodes: typing.Dict[str, OperationNode] = {}
        self._added_nodes: typing.Set[str] = set()
    
    def add_input(self, node: In):
        node = self._node_processor.process_input_node(node)
        # node.name = self._prepend_name(node.name)
        self._network.add_input(node.clone())
        self._added_nodes[node.name] = node

    def add_operation(self, node: OperationNode):
        self._operation_nodes[node.name] = node
    
    def _build_network(self, cur_node: OperationNode):

        if cur_node in self._added_nodes:
            return

        for port in cur_node.inputs:
            self._build_network(self._operation_nodes[port.module])
            
            node = self._node_processor.process_operation_node(cur_node)
            self._network.add_node(
                node.name, node.operation, node.inputs, node.labels,
                node.annotation
            )
            self._added_nodes.add(cur_node.name)
    
    def get_result(self, default_interface: NetInterface=None):
        self._network = Network()
        for name, node in self._operation_nodes.items():
            self._build_network(node)
        
        if default_interface:
            self._network.set_default_interface(default_interface)
        
        return self._network


class NameAppendVisitor(NodeVisitor):

    def __init__(self, prepend_with='', append_with=''):

        node_processor = UpdateNodeName(prepend_with, append_with)
        self._builder = NetworkBuilder(node_processor)
    
    @singledispatch
    def visit(self, node: Node):
        pass

    @visit.register
    def _(self, node: In):
        self._builder.add_input(node)

    @visit.register
    def _(self, node: OperationNode):
        self._builder.add_operation(node)

    @property
    def get_result(self, default_interface: NetInterface=None):
        return self._builder.get_result(default_interface)
    
    def reset(self):
        self._builder.reset()

    def visit_network(self, network: Network, default_interface: NetInterface=None):
        self.reset()
        network.traverse_forward(self)
        return self._builder.get_result(default_interface)


class MergeVisitor(NodeVisitor):

    def __init__(self):

        node_processor = NullNodeProcessor()
        self._builder = NetworkBuilder(node_processor)
    
    @singledispatch
    def visit(self, node: Node):
        pass

    @visit.register
    def _(self, node: In):
        self._builder.add_input(node)

    @visit.register
    def _(self, node: OperationNode):
        self._builder.add_operation(node)

    @property
    def get_result(self, default_interface: NetInterface=None):
        return self._build_network.get_result(default_interface)

    def visit_networks(
        self, networks: typing.List[Network]
    ):
        self.reset()
        input_names = []
        output_names = []

        for network in networks:
            network.traverse_forward(self)
            input_names.extend(network.input_names)
            output_names.extend(network.output_names)

        return self._builder.get_result(
            NetInterface(input_names, output_names)
        )


class ListAdapter(nn.Module):

    def __init__(self, module: nn.Module):

        self._module = module

    def forward(self, **inputs):

        return self._module.forward(inputs)


class Reorder(nn.Module):
    """Reorder the inputs when they are a list
    """
    
    def __init__(self, input_map: typing.List[int]):
        """
        Args:
            input_map (typing.List[int]): 
        """
        assert len(input_map) == len(set(input_map))
        assert max(input_map) == len(input_map) - 1
        assert min(input_map) == 0

        self._input_map = input_map
    
    def forward(self, *inputs):
        result = [None] * len(self._input_map)

        for i, v in enumerate(inputs):
            result[self._input_map[i]] = v
        
        return result


class Selector(nn.Module):
    """Select a subset of the inputs passed in.
    """

    def __init__(self, input_count: int, to_select: typing.List[int]):
        """
        Args:
            input_count (int): The number of inputs past in
            to_select (typing.List[int]): The inputs to select
        """
        assert max(to_select) <= input_count - 1
        assert min(to_select) >= 0

        self._input_count = input_count
        self._to_select = to_select

    def forward(self, *inputs):

        assert len(inputs) == self._input_count

        result = []
        for i in self._to_select:
            result.append(inputs[i])
        return result
