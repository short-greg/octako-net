import typing
import torch
from torch import nn
from torch.nn.modules import linear
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.container import Sequential

from octako.machinery.networks import ModRef, Node, OpNode, Port
from .construction2 import Args, BasicOp, ListOut, ModFactory, NullOut, diverge, Sequence, SizeOut, Sz, Var, fc
import pytest
# 1) out_size is a function
# 2) 
# fc(nn.Linear, ).out()


class TestVar:

    def test_to_var_to_var(self):
        var = Var("x")
        res = var.to(x=Var('y'))
        assert isinstance(res, Var)

    def test_to_var_to_value(self):
        var = Var("x")
        res = var.to(x=1)
        assert res == 1
    
    def test_to_var_to_val_not_contained(self):
        var = Var('x')
        res = var.to(y=2)
        assert res.name == "x"


class TestSz:

    def test_to_sz_with_valid_sz(self):

        sz = Sz(1)
        val = sz.process([torch.Size([1, 2])])
        assert val == 2

    def test_to_sz_with_invalid_sz(self):

        sz = Sz(2)
        with pytest.raises(ValueError):
            val = sz.process([torch.Size([1, 2])])

    def test_to_sz_with_valid_sz_and_port(self):

        sz = Sz(1, 0)
        val = sz.process([torch.Size([1, 2]), torch.Size([1, 2])])
        assert val == 2

    def test_to_sz_with_valid_sz_and_invalid_port(self):

        sz = Sz(1, 2)
        with pytest.raises(ValueError):
            val = sz.process([torch.Size([1, 2]), torch.Size([1, 2])])


class TestMod:

    def test_mod_with_sigmoid(self):

        m = fc(nn.Sigmoid)
        sigmoid= m.produce(torch.Size([-1, 4]))
        assert isinstance(sigmoid, nn.Sigmoid)

    def test_mod_with_nn_linear(self):

        m = fc(nn.Linear, Sz(1), 4)
        linear = m.produce(torch.Size([-1, 4]))
        assert isinstance(linear, nn.Linear)
    
    def test_mod_with_nn_linear_and_var(self):

        m = fc(nn.Linear, Sz(1), Var('x'))
        linear = m.produce(torch.Size([-1, 4]), x=3)
        assert isinstance(linear, nn.Linear)

    def test_mod_with_nn_linear_and_var(self):

        m = fc(nn.Linear, Sz(1), Var('x'))
        linear = m.produce(torch.Size([-1, 4]), x=3)
        assert isinstance(linear, nn.Linear)

    def test_op_with_nn_linear_and_var(self):

        m = fc(nn.Sigmoid).op()
        sigmoid, out_size = m.produce(torch.Size([-1, 4]), x=3)
        
        assert isinstance(sigmoid, nn.Sigmoid)


class TestOp:

    def test_mod_with_sigmoid(self):

        cur_op = fc(nn.Sigmoid).op()
        sigmoid, _ = cur_op.produce(torch.Size([-1, 4]))
        assert isinstance(sigmoid, nn.Sigmoid)

    def test_mod_with_sigmoid(self):
        
        cur_op = fc(nn.Sigmoid).op()
        nodes = list(cur_op.produce_nodes(Port("x", torch.Size([-1, 4]))))
        sigmoid = nodes[0].op
        assert isinstance(sigmoid, Sigmoid)

    def test_linear_layer(self):
        
        cur_op = fc(nn.Linear, 2, 4).op(ListOut([-1, 4]))
        nodes = list(cur_op.produce_nodes(Port("x", torch.Size([-1, 4]))))
        linear = nodes[0].op
        assert isinstance(linear, nn.Linear)


# def linear_out(self, mod: nn.Linear, in_size: torch.Size):

#     return torch.Size([
#         in_size[0],
#         mod.weight.size(0) 
#     ])



class TestSequence:

    def test_sequence_from_two_ops(self):

        sequence = (
            fc(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            fc(nn.Sigmoid).op() <<
            fc(nn.Linear, 4, 3).op(torch.Size([-1, 3]))
        )
        assert isinstance(sequence, Sequence)

    def test_sequence_produce_from_two_ops(self):

        # linear = mod(nn.Linear, Sz(1), Var('x')).op(linear_out)
        sequence, _ = (
            fc(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            fc(nn.Sigmoid).op() <<
            fc(nn.Linear, 4, 3).op(torch.Size([-1, 3]))
        ).produce([torch.Size([1, 2])])
        
        assert isinstance(sequence, Sequential)

    def test_sequence_produce_nodes_from_three_ops(self):

        # linear = mod(nn.Linear, Sz(1), Var('x')).op(linear_out)
        port = Port("mod", torch.Size([1, 2]))
        nodes = list((
            fc(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            fc(nn.Sigmoid).op() <<
            fc(nn.Linear, 4, 3).op(torch.Size([-1, 3]))
        ).produce_nodes(port))
        
        assert len(nodes) == 3

    def test_sequence_produce_nodes_from_three_ops_and_vars(self):

        # linear = mod(nn.Linear, Sz(1), Var('x')).op(linear_out)
        port = Port("mod", torch.Size([1, 2]))
        nodes = list((
            fc(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            fc(Var('activation')).op() <<
            fc(nn.Linear, 4, 3).op(torch.Size([-1, 3]))  <<
            fc(Var('activation')).op()
        ).to(activation=nn.ReLU).produce_nodes(port))
        
        assert isinstance(nodes[1].op, nn.ReLU) and isinstance(nodes[3].op, nn.ReLU)

    def test_sequence_produce_from_three_ops_and_vars(self):

        size = torch.Size([1, 2])
        linear = (
            fc(nn.Linear, Sz(1), Var('out'), bias=False).op(
                [-1, Var('x')]
            ) << 
            fc('activation').op()
        )

        sequence, _ = (
            linear.to(out=Var('x')) << 
            linear.to(out=Var('y'), activation=nn.Sigmoid)
        ).produce(size, activation=nn.ReLU, x=4, y=3)
        assert isinstance(sequence[1], nn.ReLU) and isinstance(sequence[3], nn.Sigmoid)
        assert sequence(torch.rand(1, 2)).size() == torch.Size([1, 3])

    def test_sequence_produce_from_three_ops_and_alias(self):

        size = torch.Size([1, 2])
        layer = (
            
            fc(nn.Linear, Sz(1), Var('out'), bias=False).op(
                [Sz(0), Var('out')]
            ) << 
            fc('activation').op()
        )

        sequence, sizes = (
            layer.alias(out='x') << 
            layer.to(out=Var('y'), activation=nn.Sigmoid)
        ).produce(size, activation=nn.ReLU, x=4, y=3)
        assert isinstance(sequence[1], nn.ReLU) and isinstance(sequence[3], nn.Sigmoid)
        assert sequence(torch.rand(1, 2)).size() == torch.Size([1, 3])
        assert sizes[0] == torch.Size([1, 3])


class TestDiverge:

    def test_produce_with_diverge(self):
        
        factory = diverge ([
            fc(nn.Linear, 2, 3).op(torch.Size([-1, 3])),
            fc(nn.Linear, 3, 4).op(torch.Size([-1, 4]))
        ])
        layer, _ = factory.produce([torch.Size([-1, 2]), torch.Size([-1, 3])])
        results = layer(torch.rand(2, 2), torch.ones(2, 3))
        assert results[0].size() == torch.Size([2, 3])
        assert results[1].size() == torch.Size([2, 4])

    def test_produce_nodes_with_diverge(self):

        ports = [Port(ModRef('x'), torch.Size([-1, 2])), Port(ModRef('y'), torch.Size([-1, 3]))]
        factory = diverge ([
            fc(nn.Linear, 2, 3).op(torch.Size([-1, 3])),
            fc(nn.Linear, 3, 4).op(torch.Size([-1, 4]))
        ])
        nodes: typing.List[OpNode] = []
        for node in factory.produce_nodes(ports):
            nodes.append(node)
        
        p1, = nodes[0].inputs
        p2, = nodes[1].inputs
        
        assert p1.ref.module == 'x'
        assert p2.ref.module == 'y'
