import pytest
from takonet.machinery import networks
import torch.nn as nn
import torch
from takonet.machinery.networks import Operation


class TestNode:

    def test_node_inputs_after_creation(self):

        x = networks.Port('x', torch.Size([-1, 2]))

        node = networks.OperationNode("linear", nn.Linear(2, 4),  [x], torch.Size([-1, 4]))
        assert node.inputs == [x]

    def test_node_name_after_creation(self):

        x = networks.Port('x', torch.Size([-1, 2]))
        node = networks.OperationNode("linear", nn.Linear(2, 2), [x], torch.Size([-1, 2]))
        assert node.name == 'linear'


    def test_node_forward(self):

        x = networks.Port('x', torch.Size([-1, 2]))
        node = networks.OperationNode("linear", nn.Linear(2, 2), [x], torch.Size([-1, 2]))
        assert node.forward(torch.randn(3, 2)).size() == torch.Size([3, 2])


class TestNetwork:

    def test_get_one_input_ports(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 16]))
        ])
        x, = network.get_input_ports()
        
        assert x.size == torch.Size([-1, 16])

    def test_get_two_input_ports(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 16])),
            networks.In.from_tensor('y', torch.Size([-1, 24]))
        ])
        x, y = network.get_input_ports()
        
        assert x.size == torch.Size([-1, 16])
        assert y.size == torch.Size([-1, 24])
    
    def test_add_node_ports_are_equal(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 2])),
        ])
        x, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_input_ports()
        )

        assert x.size == torch.Size([-1, 4])

    def test_output_with_one_node(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 2])),
        ])
        x, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_input_ports()
        )
        network.set_default_interface(
            network.get_input_ports(), [x]
        )
        result, = network.forward(torch.randn(3, 2))

        assert result.size() == torch.Size([3, 4])

    def test_output_with_two_nodes(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 2])),
        ])
        x, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_input_ports()
        )
        y, = network.add_node(
            'linear2', Operation(nn.Linear(4, 3), torch.Size([-1, 4])), [x], 
        )
        network.set_default_interface(
            network.get_input_ports(),
            [y]
        )
        
        result, = network.forward(torch.randn(3, 2))

        assert result.size() == torch.Size([3, 3])

    def test_get_input_names_with_one_input_is_x(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 16]))
        ])
        x, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_input_ports()
        )
        names = network.get_input_names(['linear'])
        assert names[0] == 'x'
    
    def test_are_inputs_with_real_input(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 16]))
        ])
        x, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_input_ports(),    
        )
        assert network.are_inputs(['linear'], ['x']) is True

    def test_are_inputs_with_multiple_inputs(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16])),
            networks.In.from_tensor('x2', torch.Size([-1, 16])),
        ])
        x1, x2 = network.get_input_ports()
        y1, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1],    
        )
        y2, = network.add_node(
            'linear2', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x2],    
        )
        assert network.are_inputs(['linear', 'linear2'], ['x1', 'x2']) is True

    def test_are_inputs_with_multiple_layers(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16])),
            networks.In.from_tensor('x2', torch.Size([-1, 16])),
        ])
        x1, x2 = network.get_input_ports()
        y1, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1], 
        )
        y2, = network.add_node(
            'linear2', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x2],    
        )
        z, = network.add_node(
            'linear3', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [y1, y2], 
        )
        assert network.are_inputs(['linear3'], ['x1', 'linear2']) is True


    def test_are_inputs_fails_with_single_layers(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16])),
            networks.In.from_tensor('x2', torch.Size([-1, 16])),
        ])
        x1, x2 = network.get_input_ports()
        y1, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1], 
        )
        y2, = network.add_node(
            'linear2', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x2],    
        )
        assert network.are_inputs(['linear', 'linear2'], ['x1']) is False

    def test_are_inputs_fails_with_invalid_output(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16]))
        ])
        x1,  = network.get_input_ports()
        y1, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1],    
        )
        with pytest.raises(KeyError):
            network.are_inputs(['linear3', 'linear'], ['x1']) is False


    def test_are_inputs_fails_with_invalid_input(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 16]))
        ])
        x1,  = network.get_input_ports()
        y1, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), [x1], 
            torch.Size([-1, 4])
        )
        with pytest.raises(KeyError):
            network.are_inputs(['linear'], ['x4']) is False

class Concat2(nn.Module):

    def forward(self, x, t):

        return torch.cat([x, t], dim=1)


class TestNetworkInterface:

    def test_interface_with_one_layer(self):

        network = networks.Network([
            networks.In.from_tensor('x', torch.Size([-1, 2]))
        ])
        linear, = network.add_node(
            'linear', Operation(nn.Linear(2, 4), torch.Size([-1, 4])), network.get_input_ports(), 
        )

        interface = networks.NetworkInterface(network, [linear], ['x'])
        assert interface.forward(torch.randn(3, 2)).size() == torch.Size([3, 4])


    def test_interface_with_multiple_inputs(self):

        network = networks.Network([
            networks.In.from_tensor('x1', torch.Size([-1, 2])),
            networks.In.from_tensor('x2', torch.Size([-1, 2]))
        ])
        x1, x2 = network.get_input_ports()

        concat, = network.add_node('concat', Operation(Concat2(), torch.Size([-1, 4])), [x1, x2])
        linear, = network.add_node(
            'linear', Operation(nn.Linear(4, 3), torch.Size([-1, 3])), [concat], 
        )

        interface = networks.NetworkInterface(network, [linear], ['x1', 'x2'])
        assert interface.forward(torch.randn(3, 2), torch.randn(3, 2)).size() == torch.Size([3, 3])


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
