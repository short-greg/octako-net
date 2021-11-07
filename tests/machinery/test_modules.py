import torch
import pytest
from octako.machinery import modules


class TestNetworkReorder:

    def test_reorder_with_two(self):

        reorder = modules.Reorder([1, 0])
        x, y = torch.randn(2, 1), torch.randn(2, 2)
        x1, y1 = reorder.forward(x, y)
        assert y is x1 and x is y1

    def test_reorder_with_two_of_the_same(self):

        with pytest.raises(AssertionError):
            reorder = modules.Reorder([1, 1])

    def test_reorder_with_insufficient_inputs(self):

        with pytest.raises(AssertionError):
            reorder = modules.Reorder([1, 3])
    

class TestSelector:

    def test_selector_with_two(self):

        reorder = modules.Selector(3, [1, 2])
        x, y, z = torch.randn(2, 1), torch.randn(2, 2), torch.randn(2, 2)
        x1, y1 = reorder.forward(x, y, z)
        assert y is x1 and z is y1

    def test_reorder_with_two_of_the_same(self):

        reorder = modules.Selector(2, [1, 1])
        x, y = torch.randn(2, 1), torch.randn(2, 2)
        x1, y1 = reorder.forward(x, y)
        assert y is x1 and y is y1
    
    def test_reorder_with_invalid_input(self):

        with pytest.raises(AssertionError):
            reorder = modules.Selector(3, [-1, 3])
    
    def test_reorder_fails_with_invalid_input(self):

        with pytest.raises(AssertionError):
            reorder = modules.Selector(3, [0, 4])

    def test_reorder_fails_with_improper_input_length(self):

        with pytest.raises(AssertionError):
            reorder = modules.Selector(3, [0, 2])
            x, y = torch.randn(2, 1), torch.randn(2, 2)
            reorder.forward(x, y)
