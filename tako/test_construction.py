from multiprocessing.sharedctypes import Value
from re import X
import typing
import torch
from torch import nn
from torch.nn.modules.container import Sequential
from .networks import In, InTensor, Multitap, Node, NodePort, OpNode, Out, Port
from .construction import (
    ChainFactory, CounterNamer, Meta, Kwargs, ModFactory, NetBuilder, OpFactory, OpMod, ParamMod, 
    ParameterFactory, ScalarInFactory, TensorFactory, TensorInFactory, TensorIn, TensorMod, argf, scalar_val, diverge, 
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


class TestArgf:

    def test_argf_process_with_no_special_args(self):
        v = argf(lambda x, y: x * y, [3, 2])
        res = v.process([torch.Size([1, 2])])
        assert res == 6

    def test_argf_with_two_args(self):
        v = argf(lambda x, y: x * y, [arg_.x, arg_.y])
        res = v.process([torch.Size([1, 2])], x=3, y=2)
        assert res == 6

    def test_argf_to_with_two_args(self):
        v = argf(lambda x, y: x * y, [arg_.x, arg_.y])
        v = v.to(x=3, y=2)
        res = v.process([torch.Size([1, 2])])
        assert res == 6

    def test_argf_to_with_one_arg_and_size(self):
        v = argf(lambda x, y: x * y, [arg_.x, sz[1]])
        v = v.to(x=3)
        res = v.process([torch.Size([1, 2])])
        assert res == 6

    def test_argf_procses_without_overriding(self):
        v = argf(lambda x, y: x * y, [arg_.x, arg_.y])
        v = v.to(x=3)
        with pytest.raises(ValueError):
            v.process([torch.Size([1, 2])])


class TestCounterNamer:

    def test_name_one_module(self):

        namer = CounterNamer()
        name = namer.name('X', nn.Linear(2, 2))
        assert name == 'X'

    def test_name_with_same_base_name(self):

        namer = CounterNamer()
        namer.name('X', nn.Linear(2, 2))
        name2 = namer.name('X', nn.Linear(2, 2))
        assert name2 == 'X_2'

    def test_name_with_two_different_base_names(self):

        namer = CounterNamer()
        namer.name('X', nn.Linear(2, 2))
        name = namer.name('Y', nn.Linear(2, 2))
        assert name == 'Y'

    def test_name_three_with_same_base_names(self):

        namer = CounterNamer()
        namer.name('X', nn.Linear(2, 2))
        namer.name('X', nn.Linear(2, 2))
        name = namer.name('X', nn.Linear(2, 2))
        assert name == 'X_3'

    def test_retrieve_third_name(self):

        namer = CounterNamer()
        namer.name('X', nn.Linear(2, 2))
        namer.name('X', nn.Linear(2, 2))
        namer.name('X', nn.Linear(2, 2))
        assert namer['X'][-1] == 'X_3'

    def test_retrieve_first_and_second_name(self):

        namer = CounterNamer()
        namer.name('X', nn.Linear(2, 2))
        namer.name('X', nn.Linear(2, 2))
        namer.name('X', nn.Linear(2, 2))
        assert namer['X'][:2] == ['X', 'X_2']


class TestOpMod:

    def test_op_mod_with_nn_linear(self):

        opnn = OpMod(nn)
        linear = opnn.Linear(1, 2)
        result, _ = linear.produce([Out(torch.Size([1, 1]))])
        assert isinstance(result, nn.Linear)

    def test_update_info(self):

        opnn = OpMod(nn)
        target = "Linear 1"
        linear: OpFactory = opnn.Linear(1, 2)
        result = linear.info_(name=target)
        assert result.name == target


    def test_update_labels(self):

        opnn = OpMod(nn)
        target = ['linear']
        linear: OpFactory = opnn.Linear(1, 2)
        result = linear.info_(labels=target)
        assert result.meta.labels.labels == target

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
        assert isinstance(node, InTensor)


class TestMod:

    def test_mod_with_sigmoid(self):

        m = factory(nn.Sigmoid)
        sigmoid= m.produce(Out( torch.Size([-1, 4])))[0]
        assert isinstance(sigmoid, nn.Sigmoid)

    def test_mod_with_nn_linear(self):

        m = factory(nn.Linear, sz[1], 4)
        linear = m.produce(Out(torch.Size([-1, 4])))[0]
        assert isinstance(linear, nn.Linear)
    
    def test_mod_with_nn_linear_and_arg(self):

        m = factory(nn.Linear, sz[1], arg('x'))
        linear = m.produce(Out(torch.Size([-1, 4])), x=3)[0]
        assert isinstance(linear, nn.Linear)

    def test_mod_with_nn_linear_and_arg(self):

        m = factory(nn.Linear, sz[1], arg('x'))
        linear = m.produce(Out(torch.Size([-1, 4])), x=3)[0]
        assert isinstance(linear, nn.Linear)

    def test_op_with_nn_linear_and_arg(self):

        m = factory(nn.Sigmoid)
        sigmoid, out_size = m.produce(Out(torch.Size([-1, 4])), x=3)
        
        assert isinstance(sigmoid, nn.Sigmoid)

    def test_mod_raises_error_when_arg_not_defined(self):

        m = factory(nn.Linear, sz[1], arg('x'))
        with pytest.raises(RuntimeError):
            m.produce(Out(torch.Size([-1, 4])))


class TestSequence:

    def test_sequence_from_two_ops(self):

        sequence = (
            factory(nn.Linear, 2, 4) << 
            factory(nn.Sigmoid) <<
            factory(nn.Linear, 4, 3)
        )
        assert isinstance(sequence, SequenceFactory)

    def test_sequence_produce_from_two_ops(self):
    
        sequence, _ = (
            factory(nn.Linear, 2, 4) << 
            factory(nn.Sigmoid) <<
            factory(nn.Linear, 4, 3)
        ).produce([Out(torch.Size([1, 2]))])
        
        assert isinstance(sequence, Sequential)

    def test_sequence_produce_nodes_from_three_ops(self):
        port = NodePort('mod', torch.Size([1, 2]))
        nodes = list((
            factory(nn.Linear, 2, 4) << 
            factory(nn.Sigmoid) <<
            factory(nn.Linear, 4, 3)
        ).produce_nodes(port))
        assert len(nodes) == 3

    def test_sequence_produce_nodes_from_three_ops_and_args(self):
        port = NodePort("mod", torch.Size([1, 2]))
        nodes = list((
            factory(nn.Linear, 2, 4) << 
            factory('activation') <<
            factory(nn.Linear, 4, 3)  <<
            factory('activation')
        ).to(activation=nn.ReLU).produce_nodes(port))
        
        assert isinstance(nodes[1].op, nn.ReLU) and isinstance(nodes[3].op, nn.ReLU)

    def test_update_info_for_sequential(self):

        target = 'Sequence'
        sequence = (
            factory(nn.Linear, 2, 4) << 
            factory('activation') <<
            factory(nn.Linear, 4, 3)  <<
            factory('activation')
        ).info_(name=target)
        assert sequence.name == target

    def test_sequence_produce_from_three_ops_and_args(self):

        size = torch.Size([1, 2])

        linear = (
            factory(nn.Linear, sz[1], arg('out'), bias=False) << 
            factory('activation')
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
            
            factory(nn.Linear, sz[1], arg('out'), bias=False) << 
            factory('activation')
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
            factory(nn.Linear, 2, 3),
            factory(nn.Linear, 3, 4)
        ])
        layer, _ = div.produce([Out(torch.Size([-1, 2])), Out(torch.Size([-1, 3]))])
        results = layer(torch.rand(2, 2), torch.ones(2, 3))
        assert results[0].size() == torch.Size([2, 3])
        assert results[1].size() == torch.Size([2, 4])

    def test_produce_nodes_with_diverge(self):

        ports = [NodePort('x', torch.Size([-1, 2])), NodePort('y', torch.Size([-1, 3]))]
        div = diverge ([
            factory(nn.Linear, 2, 3),
            factory(nn.Linear, 3, 4)
        ])
        nodes: typing.List[OpNode] = []
        for node in div.produce_nodes(Multitap(ports)):
            nodes.append(node)
        
        p1, = nodes[0].inputs
        p2, = nodes[1].inputs
        
        assert p1.node == 'x'
        assert p2.node == 'y'

    def test_update_info_for_diverge(self):

        div = diverge ([
            factory(nn.Linear, 2, 3),
            factory(nn.Linear, 3, 4)
        ]).info_('Diverge')
        assert div.name == 'Diverge'


class TestChain:

    def test_chained_linear(self):

        op = OpFactory(ModFactory(nn.Linear, 2, 2))
        chain_ = ChainFactory(op, [Kwargs(), Kwargs()])
        sequence, _ = chain_.produce([Out(torch.Size([-1, 2]))])

        assert isinstance(sequence[0], nn.Linear)

    def test_chained_linear_with_arg(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = ChainFactory(op, [Kwargs(x=4), Kwargs(x=5)])
        sequence, _ = chain_.produce([Out(torch.Size([-1, 2]))])

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
        for node in chain_.produce_nodes(Multitap([NodePort('x', torch.Size([-1, 2]))])):
            nodes.append(node)

        assert nodes[-1].ports[0].size == torch.Size([-1, 5])

    def test_chain_to_produce_nodes(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = chain(op, [Kwargs(x=4), Kwargs(x=5)])
        chain_ = chain_.to(x=arg('y'))
        nodes: typing.List[Node] = []
        for node in chain_.produce_nodes(NodePort("x", torch.Size([-1, 2]))):
            nodes.append(node)

        assert nodes[-1].ports[0].size == torch.Size([-1, 5])

    def test_chain_to_produce_nodes_raises_error(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = chain(op, [Kwargs(y=4), Kwargs(x=5)])
        chain_ = chain_.to(x=arg('y'))
        with pytest.raises(RuntimeError):
            for _ in chain_.produce_nodes(NodePort("x", torch.Size([-1, 2]))):
                pass

    def test_chain_(self):

        op = OpFactory(ModFactory(nn.Linear, sz[1], arg('x')))
        chain_ = chain(op, [Kwargs(y=4), Kwargs(x=5)]).info_(name='Hi')
        assert chain_.name == 'Hi'


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
            factory, name='Hi'
        )
        in_ = op.produce()
        assert isinstance(in_, InTensor)

    def test_produce_tensor_input_with_reset(self):

        factory = TensorFactory(torch.zeros, [1, 4], Kwargs())
        op = ParameterFactory(
            factory, name='Hi'
        )
        parameter = op.produce()
        assert parameter.default.size() == torch.Size([1, 4])


class TestNetBuilder:

    def test_produce_network_with_in(self):

        x = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), name='x'
        )
        op2 = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        )
        op3 = OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )
        builder = NetBuilder()
        x_, = builder.add_in(x) 
        builder.append(
            x_, op2 << op3
        )
        assert builder['x'][0].size == torch.Size([1, 2])

    def test_produce_network_with_one_op(self):

        x = TensorInFactory(
            TensorFactory(torch.randn, [1, 2])
        )
        op2 = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        )
        builder = NetBuilder()
        x, = builder.add_in(x=x)
        builder.append(
            x, op2
        )
        assert builder['Linear'][0].size == torch.Size([-1, 3])

    def test_produce_network_with_two_ops_same_name(self):

        x = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), name='x'
        )
        op2 = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        )
        op3 = OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )
        builder = NetBuilder()
        x_, = builder.add_in(x)
        builder.append(x_, op2 << op3)
        assert builder.net['Linear_2'] != builder.net['Linear']

    def test_produce_network_with_sequence(self):

        sequence = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        ) << OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )

        x = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), name='x'
        )
        
        builder = NetBuilder()
        x_ = builder.add_in(x)
        builder.append(x_, sequence)
        assert builder.net['Linear_2'] != builder.net['Linear']

    def test_produce_network_with_tensor_in(self):

        sequence = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        ) << OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )        
        
        x = TensorInFactory(
            TensorFactory(torch.randn, [1, 2]), name='x'
        )
        
        builder = NetBuilder()
        multitap = builder.add_in(x)
        builder.append(multitap, sequence)
    
        assert builder.net['Linear_2'] != builder.net['Linear']

    def test_with_multiple_net_builders(self):

        factory1 = OpFactory(
            ModFactory(nn.Linear, 2, 3)
        ) 
        
        factory2 =  OpFactory(
            ModFactory(nn.Linear, 3, 4)
        )
        
        x = TensorIn(1, 2)
        
        builder = NetBuilder()
        x_, = builder.add_in(x)
        port1, = builder.append(x_, factory1)
        port2, = builder.append(port1, factory2)
    
        net = builder.net
        y = port2.node
        y0 = port1.node
        x = x_.node
        z = net.probe([y, y0, x], by={x: torch.randn(1, 2)})
        assert z[0].size(1) == 4

    def test_output_with_chained_factories(self):

        factory1 = OpFactory(ModFactory(nn.Linear, 2, 3)) 
        factory2 =  OpFactory(ModFactory(nn.Linear, 3, 4))
        factory3 =  OpFactory(ModFactory(nn.Linear, 4, 2))
        
        x = TensorIn(1, 2)
        
        builder = NetBuilder()
        x_, = builder.add_in(x)
        port1, = builder.append(x_, factory1)
        port2, = builder.append(port1, factory2)
        port3, = builder.append(port2, factory3)
    
        net = builder.net
        y0 = port3.node
        x = x_.node
        z = net.probe([y0, x], by={x: torch.randn(1, 2)})
        assert z[0].size(1) == 2

    def test_output_with_chained_factories(self):

        factory1 = OpFactory(ModFactory(nn.Linear, sz[1], arg_.out_features)) 
        x = TensorIn(1, 2)
        
        builder = NetBuilder()
        x_, = builder.add_in(x)
        port1, = builder.append(
            x_,
            chain(factory1, [{'out_features': 3}, {'out_features': 4}, {'out_features': 2}])
        )
    
        net = builder.net
        y0 = port1.node
        x = x_.node
        z = net.probe([y0, x], by={x: torch.randn(1, 2)})
        assert z[0].size(1) == 2
