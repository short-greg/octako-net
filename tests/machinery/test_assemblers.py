from octako.machinery import assemblers
import torch.nn as nn
import torch
from octako.machinery import builders


class TestFeedForwardAssembler:

    def test_one_layer_feedforward_assembler(self):

        assembler = assemblers.DenseFeedForwardAssembler(
            builders.FeedForwardBuilder(),
            input_size=4, layer_sizes=[8], out_size=1
        )
        network = assembler.build()

        result = network.forward(torch.randn(2, 4))
        assert result[0].size() == torch.Size([2, 1])

    def test_no_layers_feedforward_assembler(self):

        assembler = assemblers.DenseFeedForwardAssembler(
            builders.FeedForwardBuilder(),
            input_size=4, layer_sizes=[], out_size=1
        )
        network = assembler.build()

        result = network.forward(torch.randn(2, 4))
        assert result[0].size() == torch.Size([2, 1])

    def test_different_activation_types(self):

        builder = builders.FeedForwardBuilder()
        builder.set_activation(builder.sigmoid)
        builder.set_out_activation(builder.tanh)
        assembler = assemblers.DenseFeedForwardAssembler(
            builder, input_size=4, layer_sizes=[3], out_size=1
        )        
        network = assembler.build()
        out_act_node = network.get_node(assembler.default_output_name)
        assert out_act_node.operation.__class__ is nn.Tanh


        act_node = network.get_node(f'activation_{0}')
        assert act_node.operation.__class__ is nn.Sigmoid

    def test_network_output_size(self):

        builder = builders.FeedForwardBuilder()
        builder.set_activation(builder.sigmoid)
        builder.set_out_activation(builder.tanh)
        assembler = assemblers.DenseFeedForwardAssembler(
            builder, input_size=4, layer_sizes=[3], out_size=1
        )        
        network = assembler.build()
        result = network.forward(torch.randn(2, 4))
        assert result[0].size() == torch.Size([2, 1])
