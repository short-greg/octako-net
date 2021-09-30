import pytest
from takonet.machinery import networks
import torch.nn as nn
import torch
from takonet.machinery.networks import In, Link, ModRef, Network, NetworkBuilder, InterfaceNode, Node, NullNodeProcessor, Operation, OpNode, Port, SubNetwork, UpdateNodeName


class TestNode:

    def test_node_inputs_after_creation(self):

        x = networks.Port('x', torch.Size([-1, 2]))

        node = networks.OpNode("lineasr", nn.Linear(2, 4),  [x], torch.Size([-1, 4]))
        assert node.inputs == [x]

    def test_node_name_after_creation(self):

        x = networks.Port('x', torch.Size([-1, 2]))
        node = networks.OpNode("linear", nn.Linear(2, 2), [x], torch.Size([-1, 2]))
        assert node.name == 'linear'


    def test_node_forward(self):

        x = networks.Port('x', torch.Size([-1, 2]))
        node = networks.OpNode("linear", nn.Linear(2, 2), [x], torch.Size([-1, 2]))
        assert node.forward(torch.randn(3, 2)).size() == torch.Size([3, 2])


class TestNetwork:

    def test_get_one_input_ports(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 16]))
        ])
        
        x, = network.get_ports('x')
        
        assert x.size == torch.Size([-1, 16])

    def test_get_two_input_ports(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 16])),
            networks.In.from_tensor('y', torch.Size([-1, 24]))
        ])
        x, y = network.get_ports(['x', 'y'])
        
        assert x.size == torch.Size([-1, 16])
        assert y.size == torch.Size([-1, 24])
    
    def test_add_node_ports_are_equal(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 2])),
        ])
        x, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_ports('x')
        )

        assert x.size == torch.Size([-1, 4])

    def test_output_with_one_node(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 2])),
        ])
        x, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_ports('x')
        )
        network.set_default_interface(
            network.get_ports('x'), [x]
        )
        result, = network.forward(torch.randn(3, 2))

        assert result.size() == torch.Size([3, 4])

    def test_output_with_two_nodes(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 2])),
        ])
        x, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_ports('x')
        )
        y, = network.add_op(
            'linear2', Operation(nn.Linear(4, 3), torch.Size([-1, 4])), [x], 
        )
        network.set_default_interface(
            network.get_ports('x'),
            [y]
        )
        
        result, = network.forward(torch.randn(3, 2))

        assert result.size() == torch.Size([3, 3])

    # def test_get_input_names_with_one_input_is_x(self):

    #     network = networks.Network([
    #         networks.In.from_tensor('x', torch.Size([-1, 16]))
    #     ])
    #     x, = network.add_op(
    #         'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_ports('x')
    #     )
    #     names = network.get_input_names(['linear'])
    #     assert names[0] == 'x'
    
    def test_are_inputs_with_real_input(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 16]))
        ])
        x, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_ports('x'),    
        )
        assert network.are_inputs(['linear'], ['x']) is True

    def test_are_inputs_with_multiple_inputs(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16])),
            networks.In.from_tensor('x2', torch.Size([-1, 16])),
        ])
        x1, x2 = network.get_ports(['x1', 'x2'])
        y1, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1],    
        )
        y2, = network.add_op(
            'linear2', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x2],    
        )
        assert network.are_inputs(['linear', 'linear2'], ['x1', 'x2']) is True

    def test_are_inputs_with_multiple_layers(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16])),
            networks.In.from_tensor('x2', torch.Size([-1, 16])),
        ])
        x1, x2 = network.get_ports(['x1', 'x2'])
        y1, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1], 
        )
        y2, = network.add_op(
            'linear2', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x2],    
        )
        z, = network.add_op(
            'linear3', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [y1, y2], 
        )
        assert network.are_inputs(['linear3'], ['x1', 'linear2']) is True

    def test_are_inputs_fails_with_single_layers(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16])),
            networks.In.from_tensor('x2', torch.Size([-1, 16])),
        ])
        x1, x2 = network.get_ports(['x1', 'x2'])
        y1, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1], 
        )
        y2, = network.add_op(
            'linear2', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x2],    
        )
        assert network.are_inputs(['linear', 'linear2'], ['x1']) is False

    def test_are_inputs_fails_with_invalid_output(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16]))
        ])
        x1,  = network.get_ports('x1')
        y1, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1],    
        )
        with pytest.raises(KeyError):
            network.are_inputs(['linear3', 'linear'], ['x1']) is False

    def test_are_inputs_fails_with_invalid_input(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16]))
        ])
        x1,  = network.get_ports('x1')
        y1, = network.add_op(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1], 
            torch.Size([-1, 4])
        )
        with pytest.raises(KeyError):
            network.are_inputs(['linear'], ['x4']) is False

class Concat2(nn.Module):

    def forward(self, x, t):

        return torch.cat([x, t], dim=1)


# Link should connect two ports...
# Need to refactor it

class TestLink:

    def test_link_to_port_from_and_to_are_correct(self):

        from_port = Port(ModRef("t"), torch.Size([2, 2]))
        to_port = Port(ModRef('h'), torch.Size([2, 2]))
        link = Link(from_port, to_port)
        assert from_port is link.from_
        assert to_port is link.to_


    def test_link_map_puts_correct_value_in_dict(self):

        from_port = Port(ModRef("t"), torch.Size([2, 2]))
        to_port = Port(ModRef('h'), torch.Size([2, 2]))
        link = Link(from_port, to_port)
        from_ = {'t': torch.randn(2, 2)}
        to_ = {}
        link.map(from_, to_)
        assert to_['h'] is from_['t']



class TestSubnetwork:

    @staticmethod
    def _setup_network():
        network = Network(
            [In('x', torch.Size([2, 2]), torch.DoubleTensor, torch.zeros(2,2)),
            In('y', torch.Size([2, 3]), torch.DoubleTensor, torch.zeros(2,2))]
        )

        x, y = network.get_ports(['x', 'y'])

        network.add_op(
            'linear1', Operation(nn.Linear(2, 3), out_size=torch.Size([-1, 3])), [x] 
        )
        
        network.add_op(
            'linear2', Operation(nn.Linear(3, 3), out_size=torch.Size([-1, 3])), [y] 
        )
        return network

    def test_probe_results_returns_correct_size(self):
        network = self._setup_network()
        x, = network.get_ports('x')
        sub_network = SubNetwork('sub', network)
        x2 = Port(ModRef('x2'), torch.Size([2, 2]))

        result = sub_network.probe(['linear1'], [Link(x2, x)], {'x2': torch.zeros(2, 2)}, True )
        assert result['linear1'].size() == torch.Size([2, 3])

    def test_get_input_ports_returns_correct_ports(self):
        network = self._setup_network()
        x, y = network.get_ports(['x', 'y'])
        sub_network = SubNetwork('sub', network)
        x2, y2 = sub_network.get_ports(['x', 'y'])
        assert x.module == x2.module
        assert y.module == y2.module

    def test_get_port_returns_correct_ports(self):
        network = self._setup_network()
        linear1,  = network.get_ports('linear1')
        sub_network = SubNetwork('sub', network)
        linear1b, = sub_network.get_ports('linear1')
        assert linear1.module == linear1b.module


class TestNetworkInterface:

    @staticmethod
    def _setup_network():
        network = Network(
            [In('x', torch.Size([2, 2]), torch.DoubleTensor, torch.zeros(2,2)),
            In('y', torch.Size([2, 3]), torch.DoubleTensor, torch.zeros(2,2))]
        )

        x, y = network.get_ports(['x', 'y'])

        network.add_op(
            'linear1', Operation(nn.Linear(2, 3), out_size=torch.Size([-1, 3])), [x] 
        )
        
        network.add_op(
            'linear2', Operation(nn.Linear(3, 3), out_size=torch.Size([-1, 3])), [y] 
        )
        return network
    
    @staticmethod
    def _setup_subnetwork(network):
        return 

    def test_probe_results_returns_correct_size(self):

        network = self._setup_network()
        sub_network = SubNetwork('sub', network)
        x, y = sub_network.get_ports(['x', 'linear1'])
        link = Link(Port(ModRef('z'), torch.Size([2, 2])), x)
        interface = InterfaceNode('interface', sub_network, [y], [link])

        result = interface.probe({'z': torch.zeros(2, 2)})
        assert result[0].size() == torch.Size([2, 3])


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


class TestNetworkReorder:

    def test_reorder_with_two(self):

        reorder = networks.Reorder([1, 0])
        x, y = torch.randn(2, 1), torch.randn(2, 2)
        x1, y1 = reorder.forward(x, y)
        assert y is x1 and x is y1

    def test_reorder_with_two_of_the_same(self):

        with pytest.raises(AssertionError):
            reorder = networks.Reorder([1, 1])

    def test_reorder_with_insufficient_inputs(self):

        with pytest.raises(AssertionError):
            reorder = networks.Reorder([1, 3])
    

class TestSelector:

    def test_selector_with_two(self):

        reorder = networks.Selector(3, [1, 2])
        x, y, z = torch.randn(2, 1), torch.randn(2, 2), torch.randn(2, 2)
        x1, y1 = reorder.forward(x, y, z)
        assert y is x1 and z is y1

    def test_reorder_with_two_of_the_same(self):

        reorder = networks.Selector(2, [1, 1])
        x, y = torch.randn(2, 1), torch.randn(2, 2)
        x1, y1 = reorder.forward(x, y)
        assert y is x1 and y is y1
    
    def test_reorder_with_invalid_input(self):

        with pytest.raises(AssertionError):
            reorder = networks.Selector(3, [-1, 3])
    
    def test_reorder_fails_with_invalid_input(self):

        with pytest.raises(AssertionError):
            reorder = networks.Selector(3, [0, 4])

    def test_reorder_fails_with_improper_input_length(self):

        with pytest.raises(AssertionError):
            reorder = networks.Selector(3, [0, 2])
            x, y = torch.randn(2, 1), torch.randn(2, 2)
            reorder.forward(x, y)
