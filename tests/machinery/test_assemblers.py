from octako.machinery import assemblers
import torch.nn as nn
import torch
from octako.machinery.networks import OpNode


class TestFeedForwardAssembler:

    def test_one_layer_feedforward_assembler(self):

        assembler = assemblers.DenseFeedForwardAssembler(
            input_size=4, layer_sizes=[8], out_size=1
        )
        network = assembler.build()

        result = network.forward(torch.randn(2, 4))
        assert result[0].size() == torch.Size([2, 1])

    def test_no_layers_feedforward_assembler(self):

        assembler = assemblers.DenseFeedForwardAssembler(
            input_size=4, layer_sizes=[], out_size=1
        )
        network = assembler.build()

        result = network.forward(torch.randn(2, 4))
        assert result[0].size() == torch.Size([2, 1])

    def test_different_activation_types(self):
        assembler = assemblers.DenseFeedForwardAssembler(
            input_size=4, layer_sizes=[3], out_size=1
        )        
        assembler.set_activation(assembler.BUILDER.sigmoid)
        assembler.set_out_activation(assembler.BUILDER.tanh)
        network = assembler.build()
        out_act_node: OpNode = network[assembler.default_output_name]
        assert out_act_node.operation.__class__ is nn.Tanh
        act_node: OpNode = network[f'activation_{0}']
        assert act_node.operation.__class__ is nn.Sigmoid

    def test_network_output_size(self):

        assembler = assemblers.DenseFeedForwardAssembler(
            input_size=4, layer_sizes=[3], out_size=1
        )     
        assembler.set_activation(assembler.BUILDER.sigmoid)
        assembler.set_out_activation(assembler.BUILDER.tanh)
        network = assembler.build()
        result = network.forward(torch.randn(2, 4))
        assert result[0].size() == torch.Size([2, 1])


class TestTargetObjectiveAssembler:

    def test_result_output_of_objective_assembler_is_correct(self):

        assembler = assemblers.TargetObjectiveAssembler(
            torch.Size([2, 4]), torch.Size([2, 4])
        )
        network = assembler.build()

        result = network.forward(torch.randn(2, 4), torch.randn(2, 4))
        assert result[0].size() == torch.Size([])

    def test_result_of_binary_cross_entropy_is_correct(self):

        assembler = assemblers.TargetObjectiveAssembler(
            torch.Size([4]), torch.Size([4])
        )
        assembler.set_objective(assembler.BUILDER.bce)
        network = assembler.build()

        result = network.forward(torch.rand(4), (torch.randn(4) > 0.0).float())
        assert result[0].size() == torch.Size([])

    def test_result_of_cross_entropy_is_correct(self):

        assembler = assemblers.TargetObjectiveAssembler(
            torch.Size([4, 4]), torch.Size([4])
        )
        assembler.set_objective(assembler.BUILDER.cross_entropy)
        network = assembler.build()

        result = network.forward(torch.log(torch.rand(4, 4)), torch.round(torch.rand(4) * 3).long())
        assert result[0].size() == torch.Size([])


class TestRegularizerObjectiveAssembler:

    def test_regularizer_objective(self):

        assembler = assemblers.RegularizerObjectiveAssembler(
            torch.Size([2, 4])
        )
        network = assembler.build()

        result = network.forward(torch.randn(2, 4))
        assert result[0].size() == torch.Size([])

    def test_result_of_binary_cross_entropy_is_correct(self):

        assembler = assemblers.RegularizerObjectiveAssembler(
            torch.Size([2, 4])
        )
        assembler.set_objective(assembler.BUILDER.l1_reg)
        network = assembler.build()

        result = network.forward(torch.randn(2, 4))
        assert result[0].size() == torch.Size([])


class TestCompoundLossAssembler:

    def test_compound_objective_with_two_losses(self):

        assembler = assemblers.TargetObjectiveAssembler(
            torch.Size([2, 4]), torch.Size([2, 4])
        )
        assembler2 = assemblers.TargetObjectiveAssembler(
            torch.Size([2, 5]), torch.Size([2, 5])
        )
        compound = assemblers.CompoundLossAssembler()
        compound.add_loss_assembler('loss1', assembler)
        compound.add_loss_assembler('loss2', assembler2)
        network = compound.build()

        result = network.forward(
            torch.randn(2, 4), torch.randn(2, 4), 
            torch.randn(2, 5), torch.randn(2, 5)
        )
        assert result[0].size() == torch.Size([])

    def test_compound_objective_with_one_loss_and_one_regularizer(self):

        assembler = assemblers.TargetObjectiveAssembler(
            torch.Size([2, 4]), torch.Size([2, 4])
        )
        assembler2 = assemblers.RegularizerObjectiveAssembler(
            torch.Size([2, 5])
        )
        assembler2.set_objective(assembler2.BUILDER.l1_reg)
        compound = assemblers.CompoundLossAssembler()
        compound.add_loss_assembler('loss', assembler)
        compound.add_regularizer_assembler('regularizer', assembler2)
        network = compound.build()

        result = network.forward(
            torch.randn(2, 4), torch.randn(2, 4), 
            torch.randn(2, 5)
        )
        assert result[0].size() == torch.Size([])

