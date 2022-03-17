from . import _networks as net
import torch
import torch.nn as nn

# by -> 


# recurrent -> probably best to have an underlying network
# recurrent.add_node() <- need to specify the step
#  

# ... adv() <-
# if a subnet is not recurrent... only need the
# outer module to be "recurrent"
#

class StepBy(net.By):
    
    # specify the limits..
    # base it on the ports in the network


    def __init__(self):
        pass

    def adv(self):
        pass


class StepPort(net.Port):

    def __init__(self, port: net.Port, step: int=0):
        self._port = port
        self._step = step

    @property
    def node(self) -> str:

        """
        Returns:
            str: Name of the node for the port
        """
        return self._port.node
    
    @property
    def port(self) -> net.Port:
        return self._port

    @property
    def step(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self._step
    
    @property
    def step(self) -> int:
        return self._step

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns:
            torch.dtype: Dtype of the port
        """
        return self._port.dtype

    @property
    def size(self) -> str:
        """
        Returns:
            torch.Size: Size of the port
        """
        return self._size

    def select(self, by: StepBy, check: bool=False):
        """
        Args:
            by (By)

        Returns:
            torch.Tensor: Output of the port for the index
        """
        result = by.get(self.node)
        if result is None:
            return None

        x = result[self.index]
        if not check:
            return x

        check_res = net.check_size(x, self.size)
        if check_res is None:
            return x
        raise ValueError(f'For mod {self._node} - index {self._index}, {check_res}')

    def select_result(self, result) -> torch.Tensor:
        """Select result of an output at the index

        Args:
            result (typing.List[torch.Tensor]): Output of a node

        Returns:
            torch.Tensor
        """
        return result[self.index]


class Recurrent(net.Network):

    def __init__(self):
        pass

    def probe(self):
        pass
