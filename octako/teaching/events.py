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

    def __init__(self, name: str=None):    
        self._listeners: typing.Dict[typing.Callable, typing.List[str]] = dict()
        self._name = name

    def add_listener(self, f: typing.Callable[[V], typing.NoReturn], listen_to_filter: typing.Union[str, typing.Set[str]]=None):
        """[Set function to call back]
        """
        if isinstance(listen_to_filter, str):
            listen_to_filter = set([listen_to_filter])
        
        if f not in self._listeners:
            self._listeners[f] = listen_to_filter

    def remove_listener(self, f: typing.Callable[[V], None]):
        """[Remove call back from listeners]
        Args:
            f (typing.Callable): [Function to call when event is fired]
        """
        assert f in self._listeners
        if f in self._listeners:
            self._listeners.pop(f)

    def invoke(self, actor: str, value: V):
        """[Call each listener]
        Args:
            value (V): [Value to update with]
        """
        for listener, listen_to_filter in self._listeners.items():
            if listen_to_filter is None or actor in listen_to_filter:
                listener(value)
