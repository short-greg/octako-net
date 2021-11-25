
import pytest
from dataclasses import dataclass

import torch
from torch._C import Size
from torch.nn.modules import dropout
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Linear
from octako.machinery.construction import UNDEFINED, ActivationFactory, AggregateFactory, BaseInput, BaseNetwork, ConvolutionFactory, DimAggregateFactory, DropoutFactory, LinearFactory, NormalizerFactory, NullFactory, RegularizerFactory, RepeatFactory, ScalerFactory, TorchLossFactory, ValidationFactory, ViewFactory
from octako.machinery.construction import check_undefined, is_undefined
from octako.machinery.networks import NetworkConstructor
from octako.modules import objectives


def test_is_undefined_is_true_when_undefined():
    x = UNDEFINED()
    assert is_undefined(x)
    
    
def test_is_undefined_is_false_when_none():
    x = None
    assert not is_undefined(x)


@dataclass
class T():
    x: int=UNDEFINED()
    a: int=1

    @check_undefined
    def as_str(self, append: str):
        return f"{self.x}_{self.a}_{append}"


def test_undefined_throws_error_when_undefined():

    t = T()
    with pytest.raises(ValueError):
        t.as_str("x")


def test_undefined_does_not_throw_error():

    t = T(x=4)
    assert t.as_str("x") == f"{t.x}_{t.a}_x"


class TestBaseNetwork:

    def test_single_port_changed_to_list(self):

        constructor = NetworkConstructor()
        in_, = constructor.add_tensor_input("x", torch.Size([-1, 4]))
        base_network = BaseNetwork(constructor, in_)
        assert isinstance(base_network.ports,list)
    
    def test_from_base_input(self):

        base_input = BaseInput(torch.Size([-1, 4]), "y")
        network = BaseNetwork.from_base_input(base_input)
        assert network.ports[0].size == base_input.size

    def test_from_base_inputs(self):

        base_inputs = [
            BaseInput(torch.Size([-1, 4]), "x"),
            BaseInput(torch.Size([-1, 4]), "y")
        ]
        network = BaseNetwork.from_base_inputs(base_inputs)
        assert network.ports[0].size == base_inputs[0].size
        assert network.ports[1].size == base_inputs[1].size


class TestTorchActivationFactory:

    def test_with_sigmoid(self):

        factory = ActivationFactory(torch.nn.Sigmoid)
        activation = factory.produce(torch.Size([-1, 4]))
        assert isinstance(activation.op, Sigmoid)
        assert activation.out_size == torch.Size([-1, 4])

    def test_with_default(self):
        factory = ActivationFactory()
        activation = factory.produce(torch.Size([-1, 4]))
        assert activation.out_size == torch.Size([-1, 4])

    def test_wtih_kwargs(self):
        factory = ActivationFactory(torch.nn.ReLU, kwargs=dict(inplace=True))
        activation = factory.produce(torch.Size([-1, 4]))
        x = torch.rand(2, 4)
        y = activation.op(x)
        assert (y == x).all()

    def test_produce_reverse(self):
        factory = ActivationFactory()
        activation = factory.produce_reverse(torch.Size([-1, 4]))
        assert activation.out_size == torch.Size([-1, 4])


class TestLinearFactory:

    def test_produce(self):

        factory = LinearFactory(4, True)
        linear = factory.produce(torch.Size([-1, 2]))
        assert linear.out_size == torch.Size([-1, 4])
        assert linear.op(torch.rand(2, 2)).size() == torch.Size([2, 4])

    def test_reverse_produce(self):
        factory = LinearFactory(4, True)
        linear = factory.produce_reverse(torch.Size([-1, 2]))
        assert linear.out_size == torch.Size([-1, 2])
        assert linear.op(torch.rand(2, 4)).size() == torch.Size([2, 2])


class TestDropoutFactory:

    def test_produce_with_default(self):

        factory = DropoutFactory()
        dropout = factory.produce(torch.Size([-1, 2]))
        assert dropout.out_size == torch.Size([-1, 2])
        assert dropout.op(torch.rand(2, 2)).size() == torch.Size([2, 2])

    def test_produce_with_alpha_dropout(self):

        factory = DropoutFactory(dropout_cls=torch.nn.AlphaDropout)
        dropout = factory.produce(torch.Size([-1, 2]))
        assert isinstance(dropout.op, torch.nn.AlphaDropout)
        assert dropout.op(torch.rand(2, 2)).size() == torch.Size([2, 2])

    def test_produce_reverse(self):
        factory = DropoutFactory()
        dropout = factory.produce_reverse(torch.Size([-1, 2]))
        assert dropout.out_size == torch.Size([-1, 2])
        assert dropout.op(torch.rand(2, 2)).size() == torch.Size([2, 2])


class TestDimAggregateFactory:

    def test_produce_with_max(self):

        factory = DimAggregateFactory(dim=1, index=0, torch_agg_fn=torch.max)
        aggregator = factory.produce(torch.Size([-1, 2]))
        assert aggregator.op(torch.rand(2, 2)).size() == torch.Size([2])
        assert aggregator.out_size == torch.Size([-1])

    def test_produce_with_mean(self):

        factory = DimAggregateFactory(dim=1, torch_agg_fn=torch.mean, keepdim=True)
        aggregator = factory.produce(torch.Size([-1, 2]))
        assert aggregator.op(torch.rand(2, 2)).size() == torch.Size([2, 1])
        assert aggregator.out_size == torch.Size([-1, 1])


class TestConvolutionFactory:

    def test_produce_with_default(self):

        factory = ConvolutionFactory(out_features=4)
        convolver = factory.produce(torch.Size([-1, 2, 4, 4]))
        assert convolver.op(torch.rand(2, 2, 4, 4)).size()[1:] == convolver.out_size[1:]

    def test_produce_with_undefined_out_features_throws_error(self):

        factory = ConvolutionFactory()
        with pytest.raises(ValueError):
            factory.produce(torch.Size([-1, 2]))

    def test_produce_reverse(self):

        in_size = torch.Size([-1, 2, 4, 4])
        factory = ConvolutionFactory(out_features=4)
        convolver = factory.produce_reverse(in_size)
        assert convolver.op(torch.rand(2, 4, 4, 4)).size()[1:] == in_size[1:]


class TestViewFactory:

    def test_produce_with_default(self):

        factory = ViewFactory(torch.Size([-1, 2, 2]))
        viewer = factory.produce(torch.Size([-1, 4]))
        assert viewer.op(torch.rand(2, 4)).size()[1:] == viewer.out_size[1:]

    def test_produce_reverse(self):

        factory = ViewFactory(torch.Size([-1, 2, 2]))
        viewer = factory.produce_reverse(torch.Size([-1, 4]))
        assert viewer.op(torch.rand(2, 2, 2)).size()[1:] == viewer.out_size[1:]


class TestRepeatFactory:

    def test_produce_with_default(self):

        factory = RepeatFactory(torch.Size([2]),keepbatch=True)
        repeater = factory.produce(torch.Size([-1, 4]))
        assert repeater.op(torch.rand(2, 4)).size()[1:] == repeater.out_size[1:]

    def test_produce_with_default_without_keepbatch(self):

        factory = RepeatFactory(torch.Size([1, 2]),keepbatch=False)
        repeater = factory.produce(torch.Size([2, 4]))
        assert repeater.op(torch.rand(2, 4)).size() == repeater.out_size


class TestNullFactory:

    def test_produce(self):

        factory = NullFactory()
        null = factory.produce(torch.Size([-1, 4]))
        assert null.op(torch.rand(2, 4)).size()[1:] == null.out_size[1:]

    def test_produce_reverse(self):

        factory = NullFactory()
        null = factory.produce_reverse(torch.Size([-1, 4]))
        assert null.op(torch.rand(2, 4)).size()[1:] == null.out_size[1:]


class TestScaleFactory:

    def test_produce(self):

        factory = ScalerFactory()
        scaler = factory.produce(torch.Size([-1, 4]))
        assert scaler.op(torch.rand(2, 4)).size()[1:] == scaler.out_size[1:]


class TestLossFactory:

    def test_produce_wtih_defaults(self):

        factory = TorchLossFactory()
        cost = factory.produce(torch.Size([-1, 4]))
        assert cost.op(torch.rand(2, 4), torch.rand(2, 4)).size() == torch.Size([])

    def test_produce_wtih_custom_loss(self):

        factory = TorchLossFactory(torch_loss_cls=torch.nn.BCELoss)
        cost = factory.produce(torch.Size([-1]))
        t =  torch.randint(0, 1, size=torch.Size([2])).float()
        assert cost.op(torch.rand(2), t).size() == torch.Size([])

    def test_produce_wtih_custom_reduction(self):

        factory = TorchLossFactory(reduction_cls=objectives.NullReduction)
        cost = factory.produce(torch.Size([-1, 4]))
        assert cost.op(torch.rand(2, 4), torch.rand(2, 4)).size()[1:] == cost.out_size[1:]


class TestRegularizerFactory:

    def test_produce_wtih_defaults(self):

        factory = RegularizerFactory()
        cost = factory.produce(torch.Size([-1, 4]))
        assert cost.op(torch.rand(2, 4)).size() == torch.Size([])

    def test_produce_wtih_custom_regularizer(self):

        factory = RegularizerFactory(regularizer_cls=objectives.L1Reg)
        cost = factory.produce(torch.Size([-1, 4]))
        assert cost.op(torch.rand(2, 4)).size() == torch.Size([])


class TestValidationFactory:

    def test_produce_wtih_defaults(self):

        factory = ValidationFactory()
        cost = factory.produce(torch.Size([-1, 4]))
        t =  torch.randint(0, 4, size=torch.Size([2])).float()
        assert cost.op(torch.rand(2, 4), t).size() == torch.Size([])

    def test_produce_wtih_custom_regularizer(self):

        factory = ValidationFactory(validation_cls=objectives.BinaryClassificationFitness)
        cost = factory.produce(torch.Size([-1, 4]))
        t =  torch.randint(0, 1, size=torch.Size([2, 4])).float()
        assert cost.op(torch.rand(2, 4), t).size() == torch.Size([])


class TestAggregateFactory:

    def test_produce_wtih_defaults(self):

        factory = AggregateFactory()
        aggregate = factory.produce(torch.Size([-1, 4]))

        assert aggregate.op(torch.rand(2, 4), torch.rand(2, 4)).size() == torch.Size([])

    def test_produce_wtih_custom_regularizer(self):

        factory = ValidationFactory(validation_cls=objectives.BinaryClassificationFitness)
        cost = factory.produce(torch.Size([-1, 4]))
        t =  torch.randint(0, 1, size=torch.Size([2, 4])).float()
        assert cost.op(torch.rand(2, 4), t).size() == torch.Size([])


