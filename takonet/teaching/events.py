import dataclasses
import typing


@dataclasses.dataclass
class IResponse:
    """[Response from a teacher]
    """

    name: str

    def summarize(self):
        return f'Name: {self.name}'


V = typing.TypeVar('V')
class TeachingEvent(typing.Generic[V]):
    """[An event fired by making progress in teaching.]
    """

    def __init__(self):    
        self._listeners = list()
        self._late_listeners = list()

    def add_listener(self, f: typing.Callable[[V], None]):
        """[Set function to call back]
        """
        if f not in self._listeners:
            self._listeners.append(f)

    # TODO: Remove
    def add_late_listener(self, f: typing.Callable[[V], None]):
        if f not in self._late_listeners:
            self._late_listeners.append(f)

    def remove_listener(self, f: typing.Callable[[V], None]):
        """[Remove call back from listeners]
        Args:
            f (typing.Callable): [Function to call when event is fired]
        """
        assert f in self._listeners
        if f in self._listeners:
            self._listeners.remove(f)

    def remove_late_listener(self, f: typing.Callable):
        assert f in self._late_listeners
        if f in self._late_listeners:
            self._late_listeners.remove(f)

    def invoke(self, value: V):
        """[Call each listener]
        Args:
            value (V): [Value to update with]
        """

        for listener in self._listeners:
            listener(value)
        
        for late_listener in self._late_listeners:
            late_listener(value)
