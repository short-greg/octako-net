import torch.nn as nn
import torch
from .networks import In, Link, ModRef, Network, InterfaceNode, Node, Operation, OpNode, Port, SubNetwork
from .visitors import NetworkBuilder, NullNodeProcessor, UpdateNodeName


class TestNullNodeProcessor:

    @staticmethod
    def _setup_node(name='name'):
        return OpNode(
            name, nn.Linear(2, 2), 
            [Port(ModRef('x'), torch.Size([-1, 2]))],
            torch.Size([-1, 2]))

    @staticmethod
    def _setup_input_node(name='name'):
        return In(
            name, torch.Size([-1, 2]), torch.Tensor, torch.randn(1, 2))

    def test_update_node_name_with_prepend(self):
        null_processor = NullNodeProcessor()
        node = self._setup_node('1')
        new_node = null_processor.process_node(node)
        assert new_node.name == node.name
        
    def test_update_node_name_with_prepend(self):
        update_node = NullNodeProcessor()
        input_node = self._setup_input_node('1')
        new_node = update_node.process_node(input_node)
        assert new_node.name == input_node.name


class TestNetworkBuilder:

    @staticmethod
    def _setup_node(input_node: In, name='name'):
        return OpNode(
            name, nn.Linear(2, 2), 
            input_node.ports,
            torch.Size([-1, 2]))

    @staticmethod
    def _setup_input_node(name='name'):
        return In(
            name, torch.Size([-1, 2]), torch.Tensor, torch.randn(1, 2))

    def test_build_network(self):
        builder = NetworkBuilder(NullNodeProcessor())
        in_ = In('in', torch.Size([-1, 2]), torch.Tensor, torch.zeros(1, 2))
        builder.add_node(in_)
        builder.add_node(self._setup_node(in_))
        network = builder.get_result()
        x, = network.get_ports('name')
        assert x.module == 'name'


class TestUpdateNodeName:

    @staticmethod
    def _setup_node(name='name'):
        return OpNode(
            name, nn.Linear(2, 2), 
            [Port(ModRef('x'), torch.Size([-1, 2]))],
            torch.Size([-1, 2]))

    def test_update_node_name_with_prepend(self):
        update_node = UpdateNodeName('x')
        node = self._setup_node('1')
        node2 = update_node.process_node(node)
        assert node2.name == 'x1'
        
    def test_update_node_name_with_prepend(self):
        update_node = UpdateNodeName(append_with='x')
        node = self._setup_node('1')
        node2 = update_node.process_node(node)
        assert node2.name == '1x'
