import pytest
import torch.nn as nn
import torch
from .networks import In, InTensor, Link, Multitap, Network, InterfaceNode, NodePort, OpNode, Out, Port, SubNetwork


class TestNode:

    def test_node_inputs_after_creation(self):

        x = NodePort('x', torch.Size([-1, 2]))

        node = OpNode('linear', nn.Linear(2, 4),  [x], Out(torch.Size([-1, 4])))
        assert node.inputs[0] == x

    def test_node_name_after_creation(self):

        x = NodePort('x', torch.Size([-1, 2]))
        node = OpNode("linear", nn.Linear(2, 2), [x], Out(torch.Size([-1, 2])))
        assert node.name == 'linear'

    def test_node_forward(self):

        x = NodePort('x', torch.Size([-1, 2]))
        node = OpNode("linear", nn.Linear(2, 2), [x], Out(torch.Size([-1, 2])))
        assert node.forward(torch.randn(3, 2)).size() == torch.Size([3, 2])

    def test_node_input_nodes_equals_x(self):

        x = NodePort('x', torch.Size([-1, 2]))
        node = OpNode("linear", nn.Linear(2, 2), [x], Out(torch.Size([-1, 2])))
        assert node.input_nodes[0] == 'x'

    def test_node_ports_equals_x(self):

        x = NodePort('x', torch.Size([-1, 2]))
        node = OpNode("linear", nn.Linear(2, 2), [x], Out(torch.Size([-1, 2])))
        assert node.inputs[0] == x

    def test_node_labels_are_correct(self):

        x = NodePort('x', torch.Size([-1, 2]))
        node = OpNode("linear", nn.Linear(2, 2), [x], Out(torch.Size([-1, 2])), labels=['linear'])
        assert node.labels == ['linear']

    def test_node_ports_is_correct_size(self):

        x = NodePort('x', torch.Size([-1, 2]))
        node = OpNode("linear", nn.Linear(2, 2), [x], Out(torch.Size([-1, 2])))
        assert node.ports[0].size == torch.Size([-1, 2])


class TestNetwork:

    def test_get_one_input_ports(self):

        network = Network([
            InTensor('x', torch.Size([-1, 16]), dtype=torch.float)
        ])
        
        x, = network['x'].ports
        
        assert x.size == torch.Size([-1, 16])

    def test_get_two_input_ports(self):

        network = Network([
            InTensor('x', torch.Size([-1, 16]), dtype=torch.float),
            InTensor('y', torch.Size([-1, 24]), dtype=torch.float)
        ])
        x, y = network[['x', 'y']].ports
        
        assert x.size == torch.Size([-1, 16])
        assert y.size == torch.Size([-1, 24])
    
    def test_add_node_ports_are_equal(self):

        network = Network([
            InTensor('x', torch.Size([-1, 2]), dtype=torch.float),
        ])
        x, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), Multitap(network['x'].ports), Out(torch.Size([-1, 4])), 
            ))

        assert x.size == torch.Size([-1, 4])

    def test_output_with_one_node(self):

        network = Network([
            InTensor('x', torch.Size([-1, 2]), dtype=torch.float),
        ])
        x, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), Multitap(network['x'].ports), Out(torch.Size([-1, 4])), 
            )
        )

        network.set_default_interface(
            network['x'].ports, [x]
        )
        result, = network.forward(torch.randn(3, 2))

        assert result.size() == torch.Size([3, 4])

    def test_output_with_two_nodes(self):

        network = Network([
            InTensor('x', torch.Size([-1, 2]), dtype=torch.float),
        ])
        x, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), Multitap(network['x'].ports), Out(torch.Size([-1, 4])), 
            ))
        
        y, = network.add_node(
            OpNode(
              'linear2', nn.Linear(4, 3), Multitap([x]), Out(torch.Size([-1, 4])), 
            ))
        network.set_default_interface(
            network['x'].ports,
            [y]
        )
        
        result, = network.forward(torch.randn(3, 2))

        assert result.size() == torch.Size([3, 3])
    
    def test_are_inputs_with_real_input(self):

        network = Network([
            InTensor('x', torch.Size([-1, 16]), dtype=torch.float)
        ])
        x, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), Multitap(network['x'].ports), Out(torch.Size([-1, 4])), 
            ))
        assert network.are_inputs(['linear'], ['x']) is True

    def test_are_inputs_with_multiple_inputs(self):

        network = Network([
            InTensor('x1', torch.Size([-1, 16]), dtype=torch.float),
            InTensor('x2', torch.Size([-1, 16]), dtype=torch.float),
        ])
        x1, x2 = network[['x1', 'x2']].ports

        y1, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), x1, Out(torch.Size([-1, 4])), 
            ))
        y2, = network.add_node(
            OpNode(
              'linear2', nn.Linear(2, 4), x2, Out(torch.Size([-1, 4])), 
            ))
        assert network.are_inputs(['linear', 'linear2'], ['x1', 'x2']) is True

    def test_are_inputs_with_multiple_layers(self):

        network = Network([
            InTensor('x1', torch.Size([-1, 16]), dtype=torch.float),
            InTensor('x2', torch.Size([-1, 16]), dtype=torch.float),
        ])
        x1, x2 = network[['x1', 'x2']].ports
        y1, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), x1, Out(torch.Size([-1, 4])), 
            ))
        y2, = network.add_node(
            OpNode(
              'linear2', nn.Linear(2, 4), x2, Out(torch.Size([-1, 4])), 
            ))
        z, = network.add_node(
            OpNode('linear3', nn.Linear(2, 4), [y1, y2], Out(torch.Size([-1, 4])))
        )
        assert network.are_inputs(['linear3'], ['x1', 'linear2']) is True

    def test_are_inputs_fails_with_single_layers(self):

        network = Network([
            InTensor('x1', torch.Size([-1, 16]), dtype=torch.float),
            InTensor('x2', torch.Size([-1, 16]), dtype=torch.float),
        ])
        x1, x2 = network[['x1', 'x2']].ports
        y1, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), x1, Out(torch.Size([-1, 4])), 
            ))
        y2, = network.add_node(
            OpNode(
              'linear2', nn.Linear(2, 4), x2, Out(torch.Size([-1, 4])), 
            ))
        assert network.are_inputs(['linear', 'linear2'], ['x1']) is True

    def test_are_inputs_fails_with_invalid_output(self):

        network = Network([
            InTensor('x1', torch.Size([-1, 16]), dtype=torch.float)
        ])
        x1,  = network['x1'].ports
        y1, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), x1, Out(torch.Size([-1, 4])), 
            ))
        with pytest.raises(KeyError):
            network.are_inputs(['linear3', 'linear'], ['x1']) is False

    def test_are_inputs_fails_with_invalid_input(self):

        network = Network([
            InTensor('x1', torch.Size([-1, 16]), torch.float)
        ])
        x1,  = network['x1'].ports
        y1, = network.add_node(
            OpNode(
              'linear', nn.Linear(2, 4), x1, Out(torch.Size([-1, 4])), 
            ))
        with pytest.raises(KeyError):
            network.are_inputs(['linear'], ['x4']) is False

class Concat2(nn.Module):

    def forward(self, x, t):

        return torch.cat([x, t], dim=1)


class TestLink:

    def test_link_to_port_from_and_to_are_correct(self):

        from_port = NodePort("t", torch.Size([2, 2]))
        to_port = NodePort('h', torch.Size([2, 2]))
        link = Link(from_port, to_port)
        assert from_port is link.from_
        assert to_port is link.to_

    def test_link_map_puts_correct_value_in_dict(self):

        from_port = NodePort("t", torch.Size([2, 2]))
        to_port = NodePort('h', torch.Size([2, 2]))
        link = Link(from_port, to_port)
        from_ = {'t': torch.randn(2, 2)}
        to_ = {}
        link.map(from_, to_)
        assert to_['h'] is from_['t']


class TestSubnetwork:

    @staticmethod
    def _setup_network():
        network = Network(
            [InTensor('x', torch.Size([2, 2]), torch.float, torch.zeros(2,2)),
            InTensor('y', torch.Size([2, 3]), torch.float, torch.zeros(2,2))]
        )

        x, y = network[['x', 'y']].ports

        network.add_node(
            OpNode(
              'linear1', nn.Linear(2, 3), x, Out(torch.Size([-1, 4])), 
            ))
        network.add_node(
            OpNode(
              'linear2', nn.Linear(2, 3), y, Out(torch.Size([-1, 4])), 
            ))
        return network

    def test_probe_results_returns_correct_size(self):
        network = self._setup_network()
        x, = network['x'].ports
        sub_network = SubNetwork('sub', network)
        x2 = NodePort('x2', torch.Size([2, 2]))

        result = sub_network.probe(
            ['linear1'], [Link(x2, x)], {'x2': torch.zeros(2, 2)}, True 
        )
        assert result[0].size() == torch.Size([2, 3])

    def test_get_input_ports_returns_correct_ports(self):
        network = self._setup_network()
        x, y = network[['x', 'y']].ports
        sub_network = SubNetwork('sub', network)
        x2, y2 = sub_network[['x', 'y']].ports
        assert x.node == x2.node
        assert y.node == y2.node

    def test_get_port_returns_correct_ports(self):
        network = self._setup_network()
        linear1,  = network['linear1'].ports
        sub_network = SubNetwork('sub', network)
        linear1b, = sub_network['linear1'].ports
        assert linear1.node == linear1b.node


class TestNetworkInterface:

    @staticmethod
    def _setup_network():
        network = Network(
            [InTensor('x', torch.Size([2, 2]), torch.float, torch.zeros(2,2)),
            InTensor('y', torch.Size([2, 3]), torch.float, torch.zeros(2,2))]
        )

        x, y = network[['x', 'y']].ports
        y1, = network.add_node(
            OpNode(
              'linear1', nn.Linear(2, 3), x, Out(torch.Size([-1, 4])), 
            ))
        y2, = network.add_node(
            OpNode(
              'linear2', nn.Linear(3, 3), y, Out(torch.Size([-1, 4])), 
            ))
        return network

    def test_probe_results_returns_correct_size(self):

        network = self._setup_network()
        sub_network = SubNetwork('sub', network)
        x, y = sub_network[['x', 'linear1']].ports
        link = Link(NodePort('z', torch.Size([2, 2])), x)
        interface = InterfaceNode('interface', sub_network, [y], [link])

        result = interface.probe({'z': torch.zeros(2, 2)})
        assert result[0].size() == torch.Size([2, 3])
