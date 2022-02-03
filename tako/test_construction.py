import typing
import torch
from torch import nn
from torch.nn.modules.container import Sequential
from .networks import In, Parameter, ModRef, Multitap, Node, OpNode, Out, Parameter, Port
from .construction import (
    ChainFactory, Info, Kwargs, ModFactory, NetBuilder, OpFactory, OpMod, ParamMod, 
    ParameterFactory, ScalarInFactory, TensorFactory, TensorInFactory, TensorIn, TensorMod, scalar_val, diverge, 
    SequenceFactory, sz, arg, factory, arg_, chain
)
import pytest


class TestArg:

    def test_to_arg_to_arg(self):
        v = arg("x")
        res = v.to(x=arg('y'))
        assert isinstance(res, arg)

    def test_to_arg_to_value(self):
        v = arg("x")
        res = v.to(x=1)
        assert res == 1
    
    def test_to_var_to_val_not_contained(self):
        v = arg('x')
        res = v.to(y=2)
        assert res.name == "x"
    
    def test_arg__creates_an_arg(self):

        v = arg_.x
        assert v.name == 'x'


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
        linear = opnn.Linear(1, 2)
        result, _ = linear.produce([Out(torch.Size([1, 1]))])
        assert isinstance(result, nn.Linear)

    def test_op_mod_with_nn_sigmoid(self):

        opnn = OpMod(nn)
        sigmoid = opnn.Sigmoid()
        result, _ = sigmoid.produce(Out(torch.Size([1, 1])))
        assert isinstance(result, nn.Sigmoid)

    def test_op_mod_with_nn_sigmoid_and_unknown_first_size(self):

        opnn = OpMod(nn)
        sigmoid = opnn.Sigmoid()
        result, _ = sigmoid.produce(Out(torch.Size([-1, 1])))
        assert isinstance(result, nn.Sigmoid)

    def test_op_mod_with_torch_tensor_and_unknown_first_size(self):

        optorch = OpMod(torch, factory=TensorMod)
        zeros = optorch.zeros(2, 3)
        # result, _ = sigmoid.produce(Out(torch.Size([-1, 1])))
        assert isinstance(zeros, TensorInFactory)

    def test_op_mod_with_torch_tensor_will_produce_in(self):

        optorch = OpMod(torch, factory=TensorMod)
        zeros = optorch.zeros(2, 3).produce()
        assert isinstance(zeros, In)

    def test_op_mod_with_torch_tensor_raises_exception_with_invalid_args(self):

        optorch = OpMod(torch, factory=TensorMod)
        with pytest.raises(RuntimeError):
            optorch.select(2, 3)

    def test_op_mod_with_torch_tensor_and_unknown_first_size(self):

        optorch = OpMod(torch, factory=TensorMod)
        with pytest.raises(RuntimeError):
            optorch.select(2, 3)

    def test_op_mod_with_param_mod(self):

        optorch = OpMod(torch, factory=ParamMod)
        rand_param = optorch.rand(2, 3)
        assert isinstance(rand_param, ParameterFactory)

    def test_op_mod_produce_with_param_mod(self):

        optorch = OpMod(torch, factory=ParamMod)
        node = optorch.rand(2, 3).produce()
        assert isinstance(node, Parameter)


class TestMod:

    def test_mod_with_sigmoid(self):

        m = factory(nn.Sigmoid)
        sigmoid= m.produce(torch.Size([-1, 4]))
        assert isinstance(sigmoid, nn.Sigmoid)

    def test_mod_with_nn_linear(self):

        m = factory(nn.Linear, sz[1], 4)
        linear = m.produce(torch.Size([-1, 4]))
        assert isinstance(linear, nn.Linear)
    
    def test_mod_with_nn_linear_and_arg(self):

        m = factory(nn.Linear, sz[1], arg('x'))
        linear = m.produce(torch.Size([-1, 4]), x=3)
        assert isinstance(linear, nn.Linear)

    def test_mod_with_nn_linear_and_arg(self):

        m = factory(nn.Linear, sz[1], arg('x'))
        linear = m.produce(torch.Size([-1, 4]), x=3)
        assert isinstance(linear, nn.Linear)

    def test_op_with_nn_linear_and_arg(self):

        m = factory(nn.Sigmoid).op()
        sigmoid, out_size = m.produce(Out(torch.Size([-1, 4])), x=3)
        
        assert isinstance(sigmoid, nn.Sigmoid)


    def test_mod_raises_error_when_arg_not_defined(self):

        m = factory(nn.Linear, sz[1], arg('x'))
        with pytest.raises(RuntimeError):
            m.produce(torch.Size([-1, 4]))

class TestSequence:

    def test_sequence_from_two_ops(self):

        sequence = (
            factory(nn.Linear, 2, 4) << 
            factory(nn.Sigmoid) <<
            factory(nn.Linear, 4, 3)
        )
        assert isinstance(sequence, SequenceFactory)

    def test_sequence_produce_from_two_ops(self):

        # linear = mod(nn.Linear, Sz(1), Var('x')).op(linear_out)
        sequence, _ = (
            factory(nn.Linear, 2, 4) << 
            factory(nn.Sigmoid).op() <<
            factory(nn.Linear, 4, 3)
        ).produce([Out(torch.Size([1, 2]))])
        
        assert isinstance(sequence, Sequential)

    def test_sequence_produce_nodes_from_three_ops(self):
        port = Port(ModRef('mod'), torch.Size([1, 2]))
        nodes = list((
            factory(nn.Linear, 2, 4).op() << 
            factory(nn.Sigmoid).op() <<
            factory(nn.Linear, 4, 3).op()
        ).produce_nodes(port))
        assert len(nodes) == 3

    def test_sequence_produce_nodes_from_three_ops_and_args(self):
        port = Port("mod", torch.Size([1, 2]))
        nodes = list((
            factory(nn.Linear, 2, 4) << 
            factory('activation') <<
            factory(nn.Linear, 4, 3).op()  <<
            factory('activation').op()
        ).to(activation=nn.ReLU).produce_nodes(port))
        
        assert isinstance(nodes[1].op, nn.ReLU) and isinstance(nodes[3].op, nn.ReLU)

    def test_sequence_produce_from_three_ops_and_args(self):

        size = torch.Size([1, 2])

        linear = (
            factory(nn.Linear, sz[1], arg('out'), bias=False).op(
                [-1, arg('x')]
            ) << 
            factory('activation').op()
        )

        sequence, _ = (
            linear.to(out=arg('x')) << 
            linear.to(out=arg('y'), activation=nn.Sigmoid)
        ).produce(Out(size), activation=nn.ReLU, x=4, y=3)
        assert isinstance(sequence[1], nn.ReLU) and isinstance(sequence[3], nn.Sigmoid)
        assert sequence(torch.rand(1, 2)).size() == torch.Size([1, 3])

    def test_sequence_produce_from_three_ops_and_alias(self):

        size = torch.Size([1, 2])
        layer = (
            
            factory(nn.Linear, sz[1], arg('out'), bias=False).op(
                [sz[0], arg('out')]
            ) << 
            factory('activation').op()
        )

        sequence, sizes = (
            layer.alias(out='x') << 
            layer.to(out=arg('y'), activation=nn.Sigmoid)
        ).produce(Out(size), activation=nn.ReLU, x=4, y=3)
        assert isinstance(sequence[1], nn.ReLU) and isinstance(sequence[3], nn.Sigmoid)
        assert sequence(torch.rand(1, 2)).size() == torch.Size([1, 3])
        assert sizes[0].size == torch.Size([-1, 3])


class TestDiverge:

    def test_produce_with_diverge(self):
        
        div = diverge ([
            factory(nn.Linear, 2, 3).op(torch.Size([-1, 3])),
            factory(nn.Linear, 3, 4).op(torch.Size([-1, 4]))
        ])
        layer, _ = div.produce([Out(torch.Size([-1, 2])), Out(torch.Size([-1, 3]))])
        results = layer(torch.rand(2, 2), torch.ones(2, 3))
        assert results[0].size() == torch.Size([2, 3])
        assert results[1].size() == torch.Size([2, 4])

    def test_produce_nodes_with_diverge(self):

        ports = [Port(ModRef('x'), torch.Size([-1, 2])), Port(ModRef('y'), torch.Size([-1, 3]))]
        div = diverge ([
            factory(nn.Linear, 2, 3).op(),
            factory(nn.Linear, 3, 4).op()
        ])
        nodes: typing.List[OpNode] = []
        for node in div.produce_nodes(Multitap(ports)):
            nodes.append(node)
        
        p1, = nodes[0].inputs
        p2, = nodes[1].inputs
        
        assert p1.ref.module == 'x'
        assert p2.ref.module == 'y'


class TestChain:

    def test_chained_linear(self):

        op = OpFactory(ModFactory(nn.Linear, 2, 2))
        chain_ = ChainFactory(op, [Kwargs(), Kwargs()])
        sequence, size = chain_.produce([Out(torch.Size([-1, 2]))])

        assert isinstance(sequence[0], nn.Linear)


    def test_chained_linear_with_arg(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = ChainFactory(op, [Kwargs(x=4), Kwargs(x=5)])
        sequence, size = chain_.produce([Out(torch.Size([-1, 2]))])

        assert isinstance(sequence[0], nn.Linear)

    def test_chained_linear_size_is_correct(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = chain(op, [Kwargs(x=4), Kwargs(x=5)])
        sequence, out = chain_.produce([Out(torch.Size([-1, 2]))])

        assert out[0].size == torch.Size([-1, 5])

    def test_chained_linear_size_raises_error_with_undefined_argument(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = chain(op, [Kwargs(y=4), Kwargs(x=5)])
        with pytest.raises(RuntimeError):
            _, out = chain_.produce([Out(torch.Size([-1, 2]))])

    def test_chained_produce_nodes(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = chain(op, [Kwargs(x=4), Kwargs(x=5)])
        nodes: typing.List[Node] = []
        for node in chain_.produce_nodes(Multitap([Port(ModRef('x'), torch.Size([-1, 2]))])):
            nodes.append(node)

        assert nodes[-1].ports[0].size == torch.Size([-1, 5])


    def test_chain_to_produce_nodes(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = chain(op, [Kwargs(x=4), Kwargs(x=5)])
        chain_ = chain_.to(x=arg('y'))
        nodes: typing.List[Node] = []
        for node in chain_.produce_nodes(Port(ModRef("x"), torch.Size([-1, 2]))):
            nodes.append(node)

        assert nodes[-1].ports[0].size == torch.Size([-1, 5])

    def test_chain_to_produce_nodes_raises_error(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = chain(op, [Kwargs(y=4), Kwargs(x=5)])
        chain_ = chain_.to(x=arg('y'))
        with pytest.raises(RuntimeError):
            for _ in chain_.produce_nodes(Port(ModRef("x"), torch.Size([-1, 2]))):
                pass


class TestTensorInFactory:

    def test_produce_tensor_input_with_call_default(self):

        factory = TensorFactory(torch.zeros,torch.Size([1, 2]), Kwargs())
        op = TensorInFactory(
            factory
        )
        in_ = op.produce()
        assert isinstance(in_, In)

    def test_produce_tensor_in_with_no_default(self):

        op = TensorIn(-1, 5, dtype=torch.float)
        in_ = op.produce()
        assert isinstance(in_, In)

    def test_produce_tensor_in_with_no_default_and_device(self):

        op = TensorIn(-1, 5, dtype=torch.float, device='cpu')
        in_ = op.produce()
        assert isinstance(in_, In)

    def test_produce_tensor_input_with_default(self):

        default = [[1, 2], [3, 4]]
        op = TensorIn(2, 2, default=default)
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

        factory = TensorFactory(torch.zeros, [1, 4], Kwargs())
        op = ParameterFactory(
            factory, Info(name='hi')
        )
        in_ = op.produce()
        assert isinstance(in_, Parameter)

    def test_produce_tensor_input_with_reset(self):

        factory = TensorFactory(torch.zeros, [1, 4], Kwargs())
        op = ParameterFactory(
            factory, Info(name='hi')
        )
        parameter = op.produce()
        parameter.reset()


class TestNetBuilder:

    def test_produce_network_with_in(self):

        op = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), info=Info(name='x')
        )
        op2 = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        )
        op3 = OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )
        builder = NetBuilder()
        builder << op << op2 << op3
        assert builder.net['x'].ports[0].size == torch.Size([1, 2])

    def test_produce_network_with_one_op(self):

        op = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), info=Info(name='x')
        )
        op2 = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        )
        builder = NetBuilder()
        builder << op << op2
        assert builder.net['Linear'].ports[0].size == torch.Size([-1, 3])

    def test_produce_network_with_two_ops_same_name(self):

        op = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), info=Info(name='x')
        )
        op2 = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        )
        op3 = OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )
        builder = NetBuilder()
        builder << op << op2 << op3
        assert builder.net['Linear_2'] != builder.net['Linear']

    def test_produce_network_with_sequence(self):

        sequence = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        ) << OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )

        op = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), info=Info(name='x')
        )
        
        builder = NetBuilder()
        builder << op << sequence
        assert builder.net['Linear_2'] != builder.net['Linear']

    def test_produce_network_with_tensor_in(self):

        sequence = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        ) << OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )        
        
        op = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), info=Info(name='x')
        )
        
        builder = NetBuilder()
        multitap = builder << op
        multitap << sequence
    
        assert builder.net['Linear_2'] != builder.net['Linear']
