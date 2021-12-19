import typing
import torch
from torch import nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.container import Sequential

from octako.machinery.networks import Node, Port
from .construction2 import Args, BasicOp, Mod, null_out, Sequence, Sz, Var, mod
import pytest
# 1) out_size is a function
# 2) 
# fc(nn.Linear, ).out()


class TestVar:

    def test_spawn_var_to_var(self):
        var = Var("x")
        res = var.spawn(x=Var('y'))
        assert isinstance(res, Var)

    def test_spawn_var_to_value(self):
        var = Var("x")
        res = var.spawn(x=1)
        assert res == 1
    
    def test_spawn_var_to_val_not_contained(self):
        var = Var('x')
        res = var.spawn(y=2)
        assert res.name == "x"


class TestSz:

    def test_spawn_sz_with_valid_sz(self):

        sz = Sz(1)
        val = sz.process([torch.Size([1, 2])])
        assert val == 2

    def test_spawn_sz_with_invalid_sz(self):

        sz = Sz(2)
        with pytest.raises(ValueError):
            val = sz.process([torch.Size([1, 2])])

    def test_spawn_sz_with_valid_sz_and_port(self):

        sz = Sz(1, 0)
        val = sz.process([torch.Size([1, 2]), torch.Size([1, 2])])
        assert val == 2

    def test_spawn_sz_with_valid_sz_and_invalid_port(self):

        sz = Sz(1, 2)
        with pytest.raises(ValueError):
            val = sz.process([torch.Size([1, 2]), torch.Size([1, 2])])


class TestMod:

    def test_mod_with_sigmoid(self):

        m = Mod(nn.Sigmoid)
        sigmoid= m.produce(torch.Size([-1, 4]))
        assert isinstance(sigmoid, nn.Sigmoid)

    def test_mod_with_nn_linear(self):

        m = Mod(nn.Linear, Sz(1), 4)
        linear = m.produce(torch.Size([-1, 4]))
        assert isinstance(linear, nn.Linear)
    
    def test_mod_with_nn_linear_and_var(self):

        m = Mod(nn.Linear, Sz(1), Var('x'))
        linear = m.produce(torch.Size([-1, 4]), x=3)
        assert isinstance(linear, nn.Linear)

    def test_mod_with_nn_linear_and_var(self):

        m = mod(nn.Linear, Sz(1), Var('x'))
        linear = m.produce(torch.Size([-1, 4]), x=3)
        assert isinstance(linear, nn.Linear)

    def test_op_with_nn_linear_and_var(self):

        m = mod(nn.Sigmoid).op()
        sigmoid, out_size = m.produce(torch.Size([-1, 4]), x=3)
        
        assert isinstance(sigmoid, nn.Sigmoid)


class TestOp:

    def test_mod_with_sigmoid(self):

        op = BasicOp(nn.Sigmoid, null_out)
        sigmoid, _ = op.produce(torch.Size([-1, 4]))
        assert isinstance(sigmoid, nn.Sigmoid)

    def test_mod_with_sigmoid(self):
        
        op = BasicOp(nn.Sigmoid, null_out)
        nodes = list(op.produce_nodes(Port("x", torch.Size([-1, 4]))))
        sigmoid = nodes[0].operation
        assert isinstance(sigmoid, Sigmoid)

    def test_linear_layer(self):
        
        op = BasicOp(nn.Linear, args=Args(2, 4), out=torch.Size([-1, 4]))
        nodes = list(op.produce_nodes(Port("x", torch.Size([-1, 4]))))
        linear = nodes[0].operation
        assert isinstance(linear, nn.Linear)


def linear_out(self, mod: nn.Linear, in_size: torch.Size):

    return torch.Size([
        in_size[0],
        mod.weight.size(0) 
    ])


class TestSequence:

    def test_sequence_from_two_ops(self):

        sequence = (
            mod(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            mod(nn.Sigmoid).op() <<
            mod(nn.Linear, 4, 3).op(torch.Size([-1, 3]))
        )
        assert isinstance(sequence, Sequence)

    def test_sequence_produce_from_two_ops(self):

        # linear = mod(nn.Linear, Sz(1), Var('x')).op(linear_out)
        sequence, _ = (
            mod(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            mod(nn.Sigmoid).op() <<
            mod(nn.Linear, 4, 3).op(torch.Size([-1, 3]))
        ).produce([torch.Size([1, 2])])
        
        assert isinstance(sequence, Sequential)
