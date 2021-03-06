from abc import ABC, abstractmethod
from functools import partial, reduce
import typing
from ._networks import Node, NodeSet, Network
from abc import ABC, abstractmethod
import typing


class Q(ABC):

    @abstractmethod
    def __call__(self, nodes: typing.Iterable[Node]) -> NodeSet:
        raise NotImplementedError
    
    def __or__(self, other):
        return OrQ([self, other])

    def __and__(self, other):
        return AndQ([self, other])

    def __sub__(self, other):
        return DifferenceQ([self, other])


class OrQ(Q):

    def __init__(self, queries: typing.List[Q]):

        self._queries = queries

    def __call__(self, nodes: typing.Iterable[Node]) -> NodeSet:
        
        node_sets = [query(nodes) for query in self._queries]
        return reduce(lambda x, y: x | y,  node_sets)

    def __or__(self, other):
        self._queries.append(other)
        return self


class AndQ(Q):

    def __init__(self, queries: typing.List[Q]):

        self._queries = queries

    def __call__(self, nodes: typing.Iterable[Node]) -> NodeSet:
        
        node_sets = [query(nodes) for query in self._queries]
        return reduce(lambda x, y: x & y,  node_sets)

    def __and__(self, other):
        self._queries.append(other)
        return self


class DifferenceQ(Q):

    def __init__(self, q1: Q, q2: Q):

        self._q1 = q1
        self._q2 = q2

    def __call__(self, nodes: typing.Iterable[Node]) -> NodeSet:
        
        return self._q1(nodes) - self._q2(nodes)


class NodeProcessor(ABC):
    
    @abstractmethod
    def process(self, node: Node) -> Node:
        pass


class UpdateNodeName(NodeProcessor):

    def __init__(self, prepend_with='', append_with=''):

        self._prepend_with = prepend_with
        self._append_with = append_with

    def __call__(self, node: Node) -> Node:
        name = self._prepend_with + node.name + self._append_with
        node = node.clone()
        node.name = name
        return node


class Filter(ABC):
    """Checks if a node should be processed or not
    """

    @abstractmethod
    def __call__(self, node) -> bool:
        pass


class NullFilter(Filter):
    """
    """

    def __call__(self, node) -> bool:
        return True


class Traverser(object):

    def __init__(self, filter: Filter=None):
        """_summary_

        Args:
            filter (Filter, optional): Filters out nodes not to process. Defaults to None.
        """

        self._filter = filter
    
    def _traverse(self, traverse_f, processor: NodeProcessor):
        for node in traverse_f():
            if not self._filter(node):
                continue
            
            processor(node)            
    
    def traverse_forward(self, net: Network, from_nodes, to_nodes, processor: NodeProcessor):
        
        self._traverse(
            partial(net.traverse_forward(from_nodes, to_nodes), processor)
        )

    def traverse_backward(self, net: Network, from_nodes, to_nodes, processor: NodeProcessor):
        
        self._traverse(
            partial(net.traverse_backward(from_nodes, to_nodes), processor)
        )         


class Backwardraverser(Traverser):

    def __init__(self, filter: Q=None):

        self._filter = filter

# class NullNodeProcessor(NodeProcessor):

#     def __init__(self):
#         pass

#     def process(self, node: Node) -> Node:
#         return node.clone()




    # TODO: move to processors tako.process
    # def _get_input_names_helper(self, node: Node, use_input: typing.List[bool], roots: typing.List):

    #     for node_input_port in node.inputs:
    #         name = node_input_port.node
    #         try:
    #             use_input[roots.index(name)] = True
    #         except ValueError:
    #             self._get_input_names_helper(self._nodes[name], use_input, roots)

    # def get_input_names(self, output_names: typing.List[str]) -> typing.List[str]:
    #     """
    #     Args:
    #         output_names (typing.List[str]): Output names in the network.
    #     Returns:
    #         typing.List[str]: The names of all the inputs required for the arguments.
    #     """
    #     use_input = [False] * len(self._roots)
    #     assert len(use_input) == len(self._roots)

    #     for output_name in output_names:
    #         if output_name not in self._nodes:
    #             raise KeyError(f'Output name {output_name} is not in the network')

    #     roots = list(self._roots)
    #     for name in output_names:
    #         self._get_input_names_helper(self._nodes[name], use_input, roots)

    #     return [input_name for input_name, to_use in zip(self._roots, use_input) if to_use is True]

    # def _is_input_name_helper(
    #     self, node: Node, input_names: typing.List[str], 
    #     is_inputs: typing.List[bool]
    # ) -> bool:
    #     """Helper to check if the node is an input for an output

    #     Args:
    #         node (Node): A node in the network
    #         input_names (typing.List[str]): Current input names
    #         is_inputs (typing.List[bool]): A list of booleans that specifies
    #         which nodes are inputs
    #     Returns:
    #         bool: Whether a node is an input
    #     """
    #     other_found = False
    #     if not node.inputs:
    #         return True

    #     for node_input_port in node.inputs:
    #         name = node_input_port.node

    #         try:
    #             is_inputs[input_names.index(name)] = True
    #         except ValueError:
    #             other_found = self._is_input_name_helper(self._nodes[name], input_names, is_inputs)
    #             if other_found: break
        
    #     return other_found

    # def are_inputs(self, output_names: typing.List[str], input_names: typing.List[str]) -> bool:
    #     """Check if a list of nodes are directly or indirectly inputs into other nodes

    #     Args:
    #         output_names (typing.List[str]): Names of nodes to check
    #         input_names (typing.List[str]): Names of input candidates

    #     Raises:
    #         KeyError: Name of the module

    #     Returns:
    #         bool: Whether or not input_names are inputs
    #     """
        
    #     is_inputs = [False] * len(input_names)

    #     for name in itertools.chain(input_names, output_names):
    #         if name not in self._nodes:
    #             raise KeyError(f'Node name {name} does not exist')
    
    #     for name in output_names:
    #         other_found: bool = self._is_input_name_helper(self._nodes[name], input_names, is_inputs)

    #         if other_found:
    #             break
    #     all_true = not (False in is_inputs)
    #     return all_true and not other_found


# TODO: Reevaluate these processors

# class NetPorts(typing.NamedTuple):

#     inputs: typing.List[typing.Union[Port, str]]
#     outputs: typing.List[typing.Union[Port, str]]


# class NetworkBuilder(object):

#     def __init__(self, node_processor: NodeProcessor):
#         self._network = None
    
#         self._nodes: typing.Dict[str, Node] = {}
#         self._sub_networks: typing.Dict[str, SubNetwork] = {}
#         self._node_processor = node_processor
#         self._added_nodes: typing.Set[str] = set()
    
#         self.reset()
    
#     def reset(self):

#         self._network = None
#         self._operation_nodes: typing.Dict[str, OpNode] = {}
#         self._network_interfaces: typing.Dict[str, InterfaceNode] = {}
#         self._added_nodes: typing.Set[str] = set()
#         self._nodes: typing.Dict[str, Node] = {}
    
#     def add_node(self, node: Node):

#         self._nodes[node.name] = node
#         return node.ports
    
#     def _build_network(self, cur_node: Node):

#         if cur_node.name in self._added_nodes:
#             return

#         for input_node in cur_node.input_nodes:
#             self._build_network(self._nodes[input_node])
            
#         node = self._node_processor.process_node(cur_node)
#         self._added_nodes.add(cur_node.name)
#         self._network.add_node(node)

#     def get_result(self, default_interface: NetPorts=None):
#         self._network = Network()
#         for name, node in self._nodes.items():
#             self._build_network(node)
        
#         if default_interface:
#             self._network.set_default_interface(default_interface)
        
#         return self._network


# class NameAppendVisitor(NodeVisitor):

#     def __init__(self, prepend_with='', append_with=''):

#         node_processor = UpdateNodeName(prepend_with, append_with)
#         self._builder = NetworkBuilder(node_processor)
    
#     def visit(self, node: Node):
#         self._builder.add_node(node)

#     @property
#     def get_result(self, default_interface: NetPorts=None):
#         return self._builder.get_result(default_interface)
    
#     def reset(self):
#         self._builder.reset()

#     def visit_network(self, network: Network, default_interface: NetPorts=None):
#         self.reset()
#         network.traverse_forward(self)

#         for sub_network in network.sub_networks:
#             sub_network.accept(self)

#         return self._builder.get_result(default_interface)


# class MergeVisitor(NodeVisitor):

#     def __init__(self):

#         node_processor = NullNodeProcessor()
#         self._builder = NetworkBuilder(node_processor)
    
#     def visit(self, node: Node):
#         self._builder.add_node(node)

#     @property
#     def get_result(self, default_interface: NetPorts=None):
#         return self._builder.get_result(default_interface)

#     def visit_networks(
#         self, networks: typing.List[Network]
#     ):
#         self.reset()
#         input_names = []
#         output_names = []

#         for network in networks:
#             network.traverse_forward(self)
#             input_names.extend(network.input_names)
#             output_names.extend(network.output_names)
        
#             for sub_network in network.sub_networks:
#                 sub_network.accept(self)

#         return self._builder.get_result(
#             NetPorts(input_names, output_names)
#         )

