import torch.nn as nn
import torch
import typing
import copy
import dataclasses
import itertools


"""
Classes related to Networks.

They are a collection of modules connected together in a graph.

"""


@dataclasses.dataclass
class Port:
    """A port into or out of a node. Used for connecting nodes together.
    """
    
    module: str
    size: torch.Size
    index: int = None

    def select(self, excitations: typing.Dict[str, torch.Tensor], index: int=None):

        excitation = excitations.get(self.module, None)
        if excitation is None: return None;

        if self.index is None:
            return excitation
        
        return excitation[self.index]


@dataclasses.dataclass
class Operation:
    """An operation performed and its output size. Used in creating the network.
    """

    op: nn.Module
    out_size: torch.Size


class Node(nn.Module):

    def __init__(
        self, name: str, 
        out_size: typing.Union[torch.Size, typing.List[torch.Size]],
        labels: typing.List[str]=None,
        annotation: str=None
    ):
        super().__init__()
        self.name: str = name
        self._out_size = out_size
        self._labels = labels
        self._annotation = annotation or ''
    
    @property
    def labels(self) -> typing.List[str]:
        return copy.copy(self._labels)
    
    @property
    def annotation(self) -> str:
        return self._annotation

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


class LabelFilter(object):

    def __init__(self, target_labels: typing.Set[str], require_all: bool=False):

        self._target_labels = target_labels
        self._filter_func = self._require_all_filter if require_all else self._require_one_filter
    
    def _require_one_filter(self, node: Node):

        return len(self._target_labels.intersection(node.labels))

    def _require_all_filter(self, node: Node):

        return len(self._target_labels.difference(node.labels)) == 0
    
    def filter(self, sequence):

        return filter(self._filter_func, sequence)


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
        super().__init__(name, out_size, labels, annotation)
        self.operation: nn.Module = operation
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
            return [Port(self.name, sz, i) for i, sz in enumerate(self._out_size)]

        return Port(self.name, self._out_size, None),
    
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
    
    def forward(self, *args):

        return self.operation(*args)


class In(Node):
    """[Input node in a network.]"""

    def __init__(self, name, sz: torch.Size, value_type: typing.Type, default_value, labels: typing.Set[str]=None, annotation: str=None):
        """[initializer]

        Args:
            name ([type]): [Name of the in node]
            out_size (torch.Size): [The size of the in node]
        """
        super().__init__(name, out_size=sz, labels=labels, annotation=annotation)
        self._value_type = value_type
        self._default_value = default_value

    def to(self, device):
        if self._value_type == torch.Tensor:
            self._default_value = self._default_value.to(device)

        self._default_value = self._default_value.to(device)

    @property
    def ports(self) -> typing.Iterable[Port]:

        return Port(self.name, self._out_size, None),
    
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
        

# TODO
# 1) Update the constructor
# 2) write in add_input
# 3) write in add_iterface
# add "scalar input" rather than tensor

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

        # for in_ in ins:
        #     if in_.name in self._nodes:
        #         raise KeyError(f'Input with name {in_.name} already exists')
        #     self._nodes[in_.name] = in_
        #     self._ins.append(in_)
        # self._ins: typing.List[In] = []
    
        self._default_ins: typing.List[str] = [] # [in_.name for in_ in self._ins]
        self._default_outs: typing.List[str] = []
        self._networks = []

    @property
    def outputs(self):
        return copy.copy(self._outs)
    
    def add_input(self, in_: In) -> typing.Iterable[Port]:
        if in_.name in self._nodes:
            raise ValueError(f'There is already a node with the name {in_.name} ')

        self._nodes[in_.name] = in_
        self._in_names.append(in_.name)
        return in_.ports

    # def add_scalar_input(self, name, data_type: typing.Type, default_value, labels: typing.Set[str]=None, annotation: str=None):
        
    #     assert isinstance(default_value, data_type) 
    #     assert name not in self._nodes
    #     in_ = ScalarIn(name, default_value, labels, annotation)
    #     self._nodes[name] = in_
    #     return in_.ports
    
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
    
    # # TODO: REMOVE
    # def set_outputs(self, ports: typing.List[Port]):

    #     self._outs = []

    #     for port in ports:
    #         if port.module not in self._nodes:
    #             raise KeyError(f'Node with name {port.module} does not exist')
    #         self._outs.append(port)
    
    def add_subnetwork(self, name, network):
        if name in self._networks:
            raise KeyError(f'Network with {name} already exists.')

        self._networks[name] = network
    
    def get_input_ports(self) -> typing.Iterable[Port]:
        ports = []

        for in_ in [self._nodes[in_name] for in_name in self._in_names]:
            
            ports.extend(in_.ports)
        return ports
    
    def get_ports(self, name) -> typing.List[Port]:

        if name not in self._nodes:
            raise KeyError(f'There is no node named {name} in the network')

        return self._nodes[name].ports
    
    def add_network_interface(
        self, name: str, network_name: str, 
        to_probe: typing.List[Port], inputs: typing.List[Port]
    ):
        interface = NetworkInterface(self._networks[network_name], ports_to_probe=to_probe)
        node = OperationNode(name, interface, inputs, [port.size for port in to_probe])
        self._nodes[name] = node.name
        return node.ports

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

        if name in self._nodes:
            raise KeyError(f'Node with name {name} already exists')

        if type(in_) == Port:
            in_ = [in_]

        for port in in_:
            assert port.module in self._nodes

        # assert (is_input and not len(inputs) > 0) or (not is_input and len(inputs) > 0)
        node = OperationNode(name, op.op, in_, op.out_size, labels)
        self._nodes[name] = node
        return node.ports
    
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
    
    def _probe_helper(
        self, node: Node, excitations: typing.Dict[str, torch.Tensor]
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
        if node.name in excitations:
            return excitations[node.name]

        inputs = []
        for port in node.inputs:
            module_name = port.module
            excitation = port.select(excitations)

            if excitation is not None:
                inputs.append(excitation)
                continue
            try:
                self._probe_helper(self._nodes[module_name], excitations)
                inputs.append(
                    port.select(excitations)
                )
            # TODO: Create a better report for this
            except KeyError:
                raise KeyError(f'Input or Node {module_name} does not exist')

        cur_result = node(*inputs)
        excitations[node.name] = cur_result
        return cur_result

    def probe(
        self, outputs: typing.List[str], by: typing.Dict[str, torch.Tensor]
    ) -> typing.List[torch.Tensor]:
        """Probe the network for its inputs

        Args:
            outputs (typing.List[str]): The nodes to probe
            by (typing.Dict[str, torch.Tensor]): The values to input into the network

        Returns:
            typing.List[torch.Tensor]: The outptus for the probe
        """

        if type(outputs) == str:
            outputs = [outputs]

        excitations = {**by}
        result  = {}

        for output in outputs:
            node = self._nodes[output]
            cur_result = self._probe_helper(node, excitations)
            result[output] = cur_result
        
        return result

    def get_node(self, key):

        return self._nodes[key]
    
    def __iter__(self):
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


class NetworkInterface(nn.Module):
    """
    """

    def __init__(
        self, network: Network, ports_to_probe: typing.List[Port], 
        by: typing.List[str]
    ):
        """An interface to the network.
        It specifies ports to probe and what to probe them by

        Args:
            network (Network): 
            ports_to_probe (typing.List[Port]): List of ports to probe
            by (typing.List[str]): The list of nodes to use as inputs. Defaults to None.  
        """

        super().__init__()
        self._network = network
        self._by = by or network.get_input_names(self._to_probe) 
        assert network.are_inputs([port.module for port in ports_to_probe], self._by)

        self._ports_to_probe: typing.List[Port] = ports_to_probe
        self._to_probe = [port.module for port in ports_to_probe]
        
        assert self._network.are_inputs(self._to_probe, self._by)
        self._input_map = {i: v for i, v in enumerate(network.get_input_names(self._to_probe))}

    def forward(self, *inputs):

        x_dict = {
            self._input_map[i]: in_ for i, in_ in enumerate(inputs)
        }
        excitations = self._network.probe(
            self._to_probe, by=x_dict
        )
        out = []
        for port in self._ports_to_probe:
            out = port.select(excitations)
        return out        
            
    def get_input_names(self):

        return self._network.get_input_names(self._to_probe)


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
