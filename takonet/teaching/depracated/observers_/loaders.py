from . import progress_bar
import typing
from src.teaching import data_controllers


class ProgressBarBuilder(object):

    def __init__(self, listen_to: typing.List[str]):
        
        self._listen_to = listen_to

    def build(self, module_accessor: data_controllers.ModuleAccessor):
        
        return progress_bar.ProgressBar(module_accessor, self._listen_to)
