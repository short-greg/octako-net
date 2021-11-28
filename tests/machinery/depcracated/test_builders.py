import pytest
from octako.machinery import builders, networks
import torch
from octako.machinery.networks import In


class TestFeedForwardBuilder:

    def test_build_dense(self):

        network = networks.Network(
            [In('x', torch.Size([-1, 2]))]
        )
        builder = builders.FeedForwardBuilder()
        x, = network.get_input_ports()
        port, = network.add_node(
            'linear', builder.dense(x.size[1], 4), x
        )
        network.set_outputs([port])
        
        x = torch.randn(3, 2)
        assert tuple(network.forward(x)[0].size()) == (3, 4)

    def test_build_dense_with_activation(self):
        network = networks.Network(
            [In('x', torch.Size([-1, 2]))]
        )
        builder = builders.FeedForwardBuilder()
        x, = network.get_input_ports()
        layer1, = network.add_node(
            'layer1', builder.activation(x.size), x
        )
        network.set_outputs([layer1])
        
        x = torch.randn(3, 2)
        assert tuple(network.forward(x)[0].size()) == (3, 2)

    def test_build_dense_with_dropout(self):
        network = networks.Network(
            [In('x', torch.Size([-1, 2]))]
        )
        builder = builders.FeedForwardBuilder()
        x_in, = network.get_input_ports()
        layer1, = network.add_node(
            'layer1', builder.dropout(x_in.size, 0.5), x_in
        )
        network.set_outputs([layer1])
        
        x = torch.randn(3, 2)
        assert tuple(network.forward(x)[0].size()) == (3, 2)

    def test_build_dense_with_normalizer(self):
        network = networks.Network(
            [In('x', torch.Size([-1, 2]))]
        )
        builder = builders.FeedForwardBuilder()
        x_in, = network.get_input_ports()
        layer1, = network.add_node(
            'layer1', builder.normalizer(x_in.size[1]), x_in
        )
        network.set_outputs([layer1])
        
        x = torch.randn(3, 2)
        assert tuple(network.forward(x)[0].size()) == (3, 2)


class TestLossBuilder:

    def test_build_mse(self):
        network = networks.Network([
            In('x', torch.Size([-1, 2])),
            In('t', torch.Size([-1, 2]))
        ])
        builder = builders.ObjectiveBuilder()
        x, t = network.get_input_ports()
        mse, = network.add_node(
            'mse', builder.mse(x.size), [x, t]
        )
        network.set_outputs([mse])
        
        x = torch.randn(3, 2)
        t = torch.randn(3, 2)
        assert network.forward(x, t)[0].size() == torch.Size([])

    def test_build_mse_non_reduction(self):
        network = networks.Network([
            In('x', torch.Size([-1, 2])),
            In('t', torch.Size([-1, 2]))
        ])
        builder = builders.ObjectiveBuilder()
        x, t = network.get_input_ports()
        mse, = network.add_node(
            'mse', builder.mse(x.size, reduction_type=builders.ReductionType.NullReduction), [x, t]
        )
        network.set_outputs([mse])
        
        x = torch.randn(3, 2)
        t = torch.randn(3, 2)
        assert network.forward(x, t)[0].size() == torch.Size([3, 2])

    def test_build_l2_regularizer(self):
        network = networks.Network(
            [In('x', torch.Size([-1, 2]))]
        )
        builder = builders.ObjectiveBuilder()
        x, = network.get_input_ports()
        reg, = network.add_node(
            'reg', builder.build_regularizer(x.size, reg_type=builders.RegularizerType.L2Reg), x
        )
        network.set_outputs([reg])
        
        x = torch.randn(3, 2)
        assert network.forward(x)[0].size() == torch.Size([])
