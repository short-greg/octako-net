# import torch.nn as nn
# import torch
# from .networks import In, Link, ModRef, Network, InterfaceNode, Node, Operation, OpNode, Port, SubNetwork
# from .visitors import NetworkBuilder, NullNodeProcessor, UpdateNodeName


# class TestNullNodeProcessor:

#     @staticmethod
#     def _setup_node(name='name'):
#         return OpNode(
#             name, nn.Linear(2, 2), 
#             [Port(ModRef('x'), torch.Size([-1, 2]))],
#             torch.Size([-1, 2]))

#     @staticmethod
#     def _setup_input_node(name='name'):
#         return In(
#             name, torch.Size([-1, 2]), torch.Tensor, torch.randn(1, 2))

#     def test_update_node_name_with_prepend(self):
#         null_processor = NullNodeProcessor()
#         node = self._setup_node('1')
#         new_node = null_processor.process_node(node)
#         assert new_node.name == node.name
        
#     def test_update_node_name_with_prepend(self):
#         update_node = NullNodeProcessor()
#         input_node = self._setup_input_node('1')
#         new_node = update_node.process_node(input_node)
#         assert new_node.name == input_node.name


# class TestNetworkBuilder:

#     @staticmethod
#     def _setup_node(input_node: In, name='name'):
#         return OpNode(
#             name, nn.Linear(2, 2), 
#             input_node.ports,
#             torch.Size([-1, 2]))

#     @staticmethod
#     def _setup_input_node(name='name'):
#         return In(
#             name, torch.Size([-1, 2]), torch.Tensor, torch.randn(1, 2))

#     def test_build_network(self):
#         builder = NetworkBuilder(NullNodeProcessor())
#         in_ = In('in', torch.Size([-1, 2]), torch.Tensor, torch.zeros(1, 2))
#         builder.add_node(in_)
#         builder.add_node(self._setup_node(in_))
#         network = builder.get_result()
#         x, = network.get_ports('name')
#         assert x.module == 'name'


# class TestUpdateNodeName:

#     @staticmethod
#     def _setup_node(name='name'):
#         return OpNode(
#             name, nn.Linear(2, 2), 
#             [Port(ModRef('x'), torch.Size([-1, 2]))],
#             torch.Size([-1, 2]))

#     def test_update_node_name_with_prepend(self):
#         update_node = UpdateNodeName('x')
#         node = self._setup_node('1')
#         node2 = update_node.process_node(node)
#         assert node2.name == 'x1'
        
#     def test_update_node_name_with_prepend(self):
#         update_node = UpdateNodeName(append_with='x')
#         node = self._setup_node('1')
#         node2 = update_node.process_node(node)
#         assert node2.name == '1x'



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