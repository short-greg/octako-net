import typing
import torch
from torch import nn
from torch.nn.modules import linear
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.container import Sequential

from octako.machinery.networks import In, ModRef, Node, OpNode, Parameter, Port
from .construction2 import Args, BasicOp, Chain, InFactory, Kwargs, ListOut, ModFactory, NetBuilder, NullOut, OpFactory, OpMod, ParameterFactory, ScalarInFactory, TensorInFactory, diverge, Sequence, SizeOut, sz, var, factory
import pytest
# 1) out_size is a function
# 2) 
# factory(nn.Linear, ).out()


class TestVar:

    def test_to_var_to_var(self):
        v = var("x")
        res = v.to(x=var('y'))
        assert isinstance(res, var)

    def test_to_var_to_value(self):
        v = var("x")
        res = v.to(x=1)
        assert res == 1
    
    def test_to_var_to_val_not_contained(self):
        v = var('x')
        res = v.to(y=2)
        assert res.name == "x"


class TestSz:

    def test_to_sz_with_valid_sz(self):

        val = sz[1].process([torch.Size([1, 2])])
        assert val == 2

    def test_to_sz_with_invalid_sz(self):

        with pytest.raises(ValueError):
            val = sz[2].process([torch.Size([1, 2])])

    def test_to_sz_with_valid_sz_and_port(self):

        val = sz[1, 0].process([torch.Size([1, 2]), torch.Size([1, 2])])
        assert val == 2

    def test_to_sz_with_valid_sz_and_invalid_port(self):

        with pytest.raises(ValueError):
            sz[1, 2].process([torch.Size([1, 2]), torch.Size([1, 2])])


class TestOpMod:

    def test_op_mod_with_nn_linear(self):

        opnn = OpMod(nn)
        linear = opnn.Linear(
            1, 2, _out=[-1, 2]
        )
        result, _ = linear.produce([torch.Size([1, 1])])
        assert isinstance(result, nn.Linear)

    def test_op_mod_with_nn_sigmoid(self):

        opnn = OpMod(nn)
        sigmoid = opnn.Sigmoid()
        result, _ = sigmoid.produce([torch.Size([1, 1])])
        assert isinstance(result, nn.Sigmoid)


class TestMod:

    def test_mod_with_sigmoid(self):

        m = factory(nn.Sigmoid)
        sigmoid= m.produce(torch.Size([-1, 4]))
        assert isinstance(sigmoid, nn.Sigmoid)

    def test_mod_with_nn_linear(self):

        m = factory(nn.Linear, sz[1], 4)
        linear = m.produce(torch.Size([-1, 4]))
        assert isinstance(linear, nn.Linear)
    
    def test_mod_with_nn_linear_and_var(self):

        m = factory(nn.Linear, sz[1], var('x'))
        linear = m.produce(torch.Size([-1, 4]), x=3)
        assert isinstance(linear, nn.Linear)

    def test_mod_with_nn_linear_and_var(self):

        m = factory(nn.Linear, sz[1], var('x'))
        linear = m.produce(torch.Size([-1, 4]), x=3)
        assert isinstance(linear, nn.Linear)

    def test_op_with_nn_linear_and_var(self):

        m = factory(nn.Sigmoid).op()
        sigmoid, out_size = m.produce(torch.Size([-1, 4]), x=3)
        
        assert isinstance(sigmoid, nn.Sigmoid)


# class TestOp:

#     def test_mod_with_sigmoid(self):

#         cur_op = factory(nn.Sigmoid).op()
#         sigmoid, _ = cur_op.produce(torch.Size([-1, 4]))
#         assert isinstance(sigmoid, nn.Sigmoid)

#     def test_mod_with_sigmoid(self):
        
#         cur_op = factory(nn.Sigmoid).op()
#         nodes = list(cur_op.produce_nodes(Port("x", torch.Size([-1, 4]))))
#         sigmoid = nodes[0].op
#         assert isinstance(sigmoid, Sigmoid)

#     def test_linear_layer(self):
        
#         cur_op = factory(nn.Linear, 2, 4).op(ListOut([-1, 4]))
#         nodes = list(cur_op.produce_nodes(Port("x", torch.Size([-1, 4]))))
#         linear = nodes[0].op
#         assert isinstance(linear, nn.Linear)


# # def linear_out(self, mod: nn.Linear, in_size: torch.Size):

# #     return torch.Size([
# #         in_size[0],
# #         mod.weight.size(0) 
# #     ])



class TestSequence:

    def test_sequence_from_two_ops(self):

        sequence = (
            factory(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            factory(nn.Sigmoid).op() <<
            factory(nn.Linear, 4, 3).op(torch.Size([-1, 3]))
        )
        assert isinstance(sequence, Sequence)

    def test_sequence_produce_from_two_ops(self):

        # linear = mod(nn.Linear, Sz(1), Var('x')).op(linear_out)
        sequence, _ = (
            factory(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            factory(nn.Sigmoid).op() <<
            factory(nn.Linear, 4, 3).op(torch.Size([-1, 3]))
        ).produce([torch.Size([1, 2])])
        
        assert isinstance(sequence, Sequential)

    def test_sequence_produce_nodes_from_three_ops(self):

        # linear = mod(nn.Linear, Sz(1), Var('x')).op(linear_out)
        port = Port("mod", torch.Size([1, 2]))
        nodes = list((
            factory(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            factory(nn.Sigmoid).op() <<
            factory(nn.Linear, 4, 3).op(torch.Size([-1, 3]))
        ).produce_nodes(port))
        assert len(nodes) == 3

    def test_sequence_produce_nodes_from_three_ops_and_vars(self):

        # linear = mod(nn.Linear, Sz(1), Var('x')).op(linear_out)
        port = Port("mod", torch.Size([1, 2]))
        nodes = list((
            factory(nn.Linear, 2, 4).op(torch.Size([-1, 4])) << 
            factory('activation').op() <<
            factory(nn.Linear, 4, 3).op(torch.Size([-1, 3]))  <<
            factory('activation').op()
        ).to(activation=nn.ReLU).produce_nodes(port))
        
        assert isinstance(nodes[1].op, nn.ReLU) and isinstance(nodes[3].op, nn.ReLU)

    
    # change factory to factory
    def test_sequence_produce_from_three_ops_and_vars(self):

        size = torch.Size([1, 2])

        linear = (
            factory(nn.Linear, sz[1], var('out'), bias=False).op(
                [-1, var('x')]
            ) << 
            factory('activation').op()
        )

        sequence, _ = (
            linear.to(out=var('x')) << 
            linear.to(out=var('y'), activation=nn.Sigmoid)
        ).produce(size, activation=nn.ReLU, x=4, y=3)
        assert isinstance(sequence[1], nn.ReLU) and isinstance(sequence[3], nn.Sigmoid)
        assert sequence(torch.rand(1, 2)).size() == torch.Size([1, 3])

    def test_sequence_produce_from_three_ops_and_alias(self):

        size = torch.Size([1, 2])
        layer = (
            
            factory(nn.Linear, sz[1], var('out'), bias=False).op(
                [sz[0], var('out')]
            ) << 
            factory('activation').op()
        )

        sequence, sizes = (
            layer.alias(out='x') << 
            layer.to(out=var('y'), activation=nn.Sigmoid)
        ).produce(size, activation=nn.ReLU, x=4, y=3)
        assert isinstance(sequence[1], nn.ReLU) and isinstance(sequence[3], nn.Sigmoid)
        assert sequence(torch.rand(1, 2)).size() == torch.Size([1, 3])
        assert sizes[0] == torch.Size([1, 3])


class TestDiverge:

    def test_produce_with_diverge(self):
        
        div = diverge ([
            factory(nn.Linear, 2, 3).op(torch.Size([-1, 3])),
            factory(nn.Linear, 3, 4).op(torch.Size([-1, 4]))
        ])
        layer, _ = div.produce([torch.Size([-1, 2]), torch.Size([-1, 3])])
        results = layer(torch.rand(2, 2), torch.ones(2, 3))
        assert results[0].size() == torch.Size([2, 3])
        assert results[1].size() == torch.Size([2, 4])

    def test_produce_nodes_with_diverge(self):

        ports = [Port(ModRef('x'), torch.Size([-1, 2])), Port(ModRef('y'), torch.Size([-1, 3]))]
        div = diverge ([
            factory(nn.Linear, 2, 3).op(torch.Size([-1, 3])),
            factory(nn.Linear, 3, 4).op(torch.Size([-1, 4]))
        ])
        nodes: typing.List[OpNode] = []
        for node in div.produce_nodes(ports):
            nodes.append(node)
        
        p1, = nodes[0].inputs
        p2, = nodes[1].inputs
        
        assert p1.ref.module == 'x'
        assert p2.ref.module == 'y'


class TestChain:

    def test_chained_linear(self):

        op = BasicOp(ModFactory(nn.Linear, 2, 2), out=[-1, 2])
        chain = Chain(op, [Kwargs(), Kwargs()])
        sequence, size = chain.produce([torch.Size([-1, 2])])

        assert isinstance(sequence[0], nn.Linear)


    def test_chained_linear_with_var(self):

        op = BasicOp(ModFactory(nn.Linear, sz[1], var('x')), out=[-1, var('x')])
        chain = Chain(op, [Kwargs(x=4), Kwargs(x=5)])
        sequence, size = chain.produce([torch.Size([-1, 2])])

        assert isinstance(sequence[0], nn.Linear)

    def test_chained_linear_size_is_correct(self):

        op = BasicOp(ModFactory(nn.Linear, sz[1], var('x')), out=[-1, var('x')])
        chain = Chain(op, [Kwargs(x=4), Kwargs(x=5)])
        sequence, size = chain.produce([torch.Size([-1, 2])])

        assert size[0] == torch.Size([-1, 5])

    def test_chained_produce_nodes(self):

        op = BasicOp(ModFactory(nn.Linear, sz[1], var('x')), out=[-1, var('x')])
        chain = Chain(op, [Kwargs(x=4), Kwargs(x=5)])
        nodes: typing.List[Node] = []
        for node in chain.produce_nodes(Port("x", torch.Size([-1, 2]))):
            nodes.append(node)

        assert nodes[-1].ports[0].size == torch.Size([-1, 5])


    def test_chain_to_produce_nodes(self):

        op = BasicOp(ModFactory(nn.Linear, sz[1], var('x')), out=[-1, var('x')])
        chain = Chain(op, [Kwargs(x=4), Kwargs(x=5)])
        chain = chain.to(x=var('y'))
        nodes: typing.List[Node] = []
        for node in chain.produce_nodes(Port("x", torch.Size([-1, 2]))):
            nodes.append(node)

        assert nodes[-1].ports[0].size == torch.Size([-1, 5])


class TestTensorInFactory:

    def test_produce_tensor_input_with_call_default(self):

        op = TensorInFactory(
            torch.Size([1, 2]), torch.ones, True
        )
        in_ = op.produce()
        assert isinstance(in_, In)


    def test_produce_tensor_input_with_no_call(self):

        op = TensorInFactory(
            torch.Size([1, 2]), torch.tensor([[2, 3]]), False
        )
        in_ = op.produce()
        assert isinstance(in_, In)


class TestScalarInFactory:

    def test_produce_tensor_input(self):

        op = ScalarInFactory(
            int, 2, False
        )
        in_ = op.produce()

        assert isinstance(in_, In)

    def test_produce_tensor_input_with_call_default(self):

        op = ScalarInFactory(
            type(dict()), dict, True
        )
        in_ = op.produce()

        assert isinstance(in_, In)


class TestParameterFactory:

    def test_produce_tensor_input_with_call_default(self):

        op = ParameterFactory(
            torch.Size([1, 2]), torch.ones
        )
        in_ = op.produce()
        assert isinstance(in_, Parameter)

    def test_produce_tensor_input_with_reset(self):

        op = ParameterFactory(
            torch.Size([1, 2]), torch.ones
        )
        parameter = op.produce()
        parameter.reset()


class TestNetBuilder:

    def test_produce_simple_network(self):

        op = TensorInFactory(
            torch.Size([1, 2]), torch.ones, True
        )
        op2 = BasicOp(
            ModFactory(nn.Linear, 2, 3), out=[-1, 3]
        )
        op3 = BasicOp(
            ModFactory(nn.Linear, 3, 4), out=[-1, 4]
        )
        builder = NetBuilder()
        builder << op << op2 << op3

