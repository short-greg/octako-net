from . networks import NodeVisitor, Node, Port, Network, SubNetwork, OpNode, InterfaceNode
from abc import ABC, abstractmethod
import typing



class NodeProcessor(ABC):
    
    @abstractmethod
    def process_node(self, node: Node) -> Node:
        pass


class UpdateNodeName(NodeProcessor):

    def __init__(self, prepend_with='', append_with=''):

        self._prepend_with = prepend_with
        self._append_with = append_with

    def _update_name(self, node: Node):

        name = self._prepend_with + node.name + self._append_with
        node = node.clone()
        node.name = name
        return node

    def process_node(self, node: Node) -> Node:
        return self._update_name(node)


class NullNodeProcessor(NodeProcessor):

    def __init__(self):
        pass

    def process_node(self, node: Node) -> Node:
        return node.clone()


class NetPorts(typing.NamedTuple):

    inputs: typing.List[typing.Union[Port, str]]
    outputs: typing.List[typing.Union[Port, str]]


class NetworkBuilder(object):

    def __init__(self, node_processor: NodeProcessor):
        self._network = None
    
        self._nodes: typing.Dict[str, Node] = {}
        self._sub_networks: typing.Dict[str, SubNetwork] = {}
        self._node_processor = node_processor
        self._added_nodes: typing.Set[str] = set()
    
        self.reset()
    
    def reset(self):

        self._network = None
        self._operation_nodes: typing.Dict[str, OpNode] = {}
        self._network_interfaces: typing.Dict[str, InterfaceNode] = {}
        self._added_nodes: typing.Set[str] = set()
        self._nodes: typing.Dict[str, Node] = {}
    
    def add_node(self, node: Node):

        self._nodes[node.name] = node
        return node.ports
    
    def _build_network(self, cur_node: Node):

        if cur_node.name in self._added_nodes:
            return

        for input_node in cur_node.input_nodes:
            print(input_node)
            self._build_network(self._nodes[input_node])
            
        node = self._node_processor.process_node(cur_node)
        self._added_nodes.add(cur_node.name)
        self._network.add_node(node)

    def get_result(self, default_interface: NetPorts=None):
        self._network = Network()
        for name, node in self._nodes.items():
            print('Building ', node.name)
            self._build_network(node)
        
        if default_interface:
            self._network.set_default_interface(default_interface)
        
        return self._network


class NameAppendVisitor(NodeVisitor):

    def __init__(self, prepend_with='', append_with=''):

        node_processor = UpdateNodeName(prepend_with, append_with)
        self._builder = NetworkBuilder(node_processor)
    
    def visit(self, node: Node):
        self._builder.add_node(node)

    @property
    def get_result(self, default_interface: NetPorts=None):
        return self._builder.get_result(default_interface)
    
    def reset(self):
        self._builder.reset()

    def visit_network(self, network: Network, default_interface: NetPorts=None):
        self.reset()
        network.traverse_forward(self)

        for sub_network in network.sub_networks:
            sub_network.accept(self)

        return self._builder.get_result(default_interface)


class MergeVisitor(NodeVisitor):

    def __init__(self):

        node_processor = NullNodeProcessor()
        self._builder = NetworkBuilder(node_processor)
    
    def visit(self, node: Node):
        self._builder.add_node(node)

    @property
    def get_result(self, default_interface: NetPorts=None):
        return self._builder.get_result(default_interface)

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
        
            for sub_network in network.sub_networks:
                sub_network.accept(self)

        return self._builder.get_result(
            NetPorts(input_names, output_names)
        )
