from abc import abstractmethod
import typing
from sango.nodes import STORE_REF, Action, Conditional, Status, Task, TaskDecorator, TickDecorator, TickDecorator2nd, Tree, action, cond, decorate, loads_, loads, task, task_, until, var
from sango.vars import Const, Ref, Var, ref
from torch.types import Storage
from torch.utils.data.dataset import Dataset
from octako.machinery.networks import Network

from octako.modules import Parallel
from .construction import Sequence
from .learners import Learner, Tester
from tqdm import tqdm
from functools import partial
from dataclasses import dataclass, is_dataclass
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Union



@dataclass
class Progress:

    name: str
    total_epochs: int
    cur_epoch: int = 0
    total_iterations: int = 0 
    cur_iteration: int = 0


class ProgressRecorder(object):

    def __init__(self, default_epochs: int=1):
        self._progresses = {}
        self._cur_progress: str = None
        self._default_epochs = default_epochs

    def add(self, name: str, n_epochs: int=None, total_iterations: int=0, switch=True):
        if name not in self._progresses:
            raise ValueError(f'Progress named {name} already exists.')
        
        n_epochs = n_epochs if n_epochs is not None else self._default_epochs

        self._progresses[name] = Progress(
            name, n_epochs, total_iterations=total_iterations
        )
        if switch: self.switch(name)

    def switch(self, name: str):
        if name not in self._progresses:
            raise ValueError(f'Progress named {name} does not exist.')
        
        self._cur_progress = name
    
    def get(self, name: str):
        return self._progresses[name]
    
    def names(self):
        return list(self._progresses.keys())

    @property
    def cur(self) -> Progress:
        return self._progresses[self._cur_progress]
    
    def complete(self):
        self._completed = True
    
    def adv_epoch(self, total_iterations=0):
        self.cur.cur_epoch += 1
        self.cur.total_iterations = total_iterations

    def adv_iter(self):
        self.cur.cur_iteration += 1


class Results:
    
    def __init__(self):
        
        self.df = pd.DataFrame()
        self._progress_cols = set()
        self._result_cols = set
    
    def add_result(self, teacher: str, progress: Progress, results: typing.Dict[str, float]):

        self._progress_cols.update(
            progress.to_dict().keys()
        )

        self._result_cols.update(
            results.keys()
        )

        self.df.loc[len[self.df]] = {
            self.teacher_col: teacher,
            **progress.to_dict(),
            **results
        }
    
    @property
    def teacher_col(self):
        return "Teacher"
    
    @property
    def result_cols(self):
        return set(*self._result_cols)
    
    @property
    def progress_cols(self):
        return set(*self._progress_cols)


class Teach(Action):
    
    result = Var()

    def __init__(
        self, name: str, learner: Const[Union[Learner, Tester]], 
        dataset: Const[Dataset], 
        results: Const[Results],
        progress: Const[ProgressRecorder],
        batch_size: Const[int]
    ):
        
        self._name = name
        self._learner = learner
        self._dataset = dataset
        self._iter = None
        self._results = results
        self._progress = progress
        self._batch_size = batch_size

    def _setup_progress(self):
        self._iter = DataLoader(
            self._dataset.val, self._batch_size, shuffle=True
        )
        n_iterations = len(self._iter)
        if self._name in self._progress:
            self._progress.val.switch(self._name)
            self._progress.val.adv_epoch(n_iterations)
        else:
            self._progress.val.add(
                self._name, total_iterations=n_iterations, switch=True
            )
    
    def _setup_iter(self):

        is_setup = self._iter is not None
        if not is_setup:
            self._setup_progress()
        else:
            self._progress.val.adv_iter()

    @abstractmethod
    def perform_action(self, x, t):
        pass

    def act(self):
        self._setup_iter()

        try:
            x, t = next(self._iter)
        except StopIteration:
            self._iter = None
            return Status.SUCCESS
        
        result = self.perform_action(x, t)

        self._results.val.store(self._name, self._progress.cur, result)
        return Status.RUNNING


class Train(Teach):

    def perform_action(self, x, t):
        return self._learner.val.learn(x, t)


class Validate(Teach):
    
    def perform_action(self, x, t):
        return self._learner.val.test(x, t)


class Trainer(Tree):

    n_batches = var(10)
    batch_size = var(32)
    validation_dataset = var()
    training_dataset = var()
    network = var()

    @task
    class entry(Parallel):
        update_progress = action('update_progress_bar')

        @task
        class train(Sequence):
            execute = action('execute', STORE_REF)
            class epoch(Sequence):
                train = task_(
                    Train, 'Trainer', ref.learner, 
                    ref.training_dataset, ref.results, 
                    ref.batch_size
                )
                validate = task_(
                    Validate, 'Validator', 
                    ref.learner, ref.validation_dataset, 
                    ref.results, ref.batch_size
                )


    def __init__(self, name: str, network: Network):
        super().__init__(name)
        self._progress = ProgressRecorder()
        self._network.value = network

    def load_datasets(self):
        pass

    def execute(self, store: Storage):
        cur_batch = store.get_or_create('cur_batch', 0)

        if cur_batch.val == self._n_batches:
            return Status.FAILURE

        cur_batch.val += 1
        self._progress.complete()

        return Status.SUCCESS

    def update_progress_bar(self, store: Storage):
        
        pbar = store.get_or_create('pbar')

        if self._progress.completed:
            if not pbar.empty(): pbar.close()
            return Status.SUCCESS
        
        if pbar.is_empty():
            pbar.val = tqdm(total=self._progress.cur.total_iterations)
        pbar.set_description_str(self._progress.cur.name)
        pbar.update(1)

        # self.pbar.total = lecture.n_lesson_iterations
        # self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
        # pbar.refresh()
        return Status.RUNNING


class LoadDatasets(Action):
    
    training_dataset = var()
    validation_dataset = var()
    material = var()

    def __init__(self, train_for_all: bool):

        self._train_for_all = train_for_all

    def reset(self):

        pass

    def act(self):
        
        # load the material
        pass



# def progress_bar(iterations: int):

#     pbar: tqdm = None
#     final_status = None

#     def _(node: Task):
    
#         nonlocal iterations
#         if isinstance(iterations, Ref):
#             iterations = iterations.shared(node._storage)

#         def tick(node: Action, wrapped_tick):
#             nonlocal pbar
#             nonlocal final_status
#             if pbar is None:
#                 pbar = tqdm(total=iterations)
#                 pbar.set_description_str(f'{node.name}')
#                 pbar.update(1)
#                 # TODO: How to get the results ub
#                 # self.pbar.total = lecture.n_lesson_iterations
#                 # self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
#                 pbar.refresh()
#                 return Status.RUNNING
            
#             if final_status is None:

#                 status = wrapped_tick()
#                 if status.done:
#                     final_status = status
#                 return Status.RUNNING

#             pbar.close()
#             return final_status
        
#         return TickDecorator(node, tick)
#     return _



# def neg(node: Task):

#     def tick(node: Task, wrapped_tick):
#         status = wrapped_tick()
#         if status == Status.SUCCESS:
#             return Status.FAILURE
#         elif status == Status.FAILURE:
#             return Status.SUCCESS
#         return status
    
#     return TickDecorator(node, tick)

# def upto(iterations: int):

#     def _(node: Task):
    
#         i = 0
#         nonlocal iterations
#         if isinstance(iterations, Ref):
#             iterations = iterations.shared(node._storage)

#         def tick(node: Action, wrapped_tick):
#             nonlocal i
#             if i >= iterations:
#                 return Status.DONE
#             result = wrapped_tick()
#             if result == Status.FAILURE:
#                 return result
            
#             i += 1
#             if i == iterations:
#                 return Status.SUCCESS
#             return Status.RUNNING
        
#         return TickDecorator(node, tick)
#     return _


# def progress_bar_(iterations: int):

#     return partial(progress_bar, iterations=iterations)



# class Trainer(Tree):

#     n_batches = var(10)
#     batch_size = var(32)
#     progress = var()
#     material = var()

#     class entry(Sequence):
        
#         # variables
#         training_dataset = var()
#         validation_dataset = var()

#         # tasks
#         load_datasets = task_(LoadDatasets, Ref('material'))

#         @upto(Ref('n_batches'))
#         class train(Sequence):
#             train = (
#                 loads_(progress_bar, ref.progress, ref.training) <<
#                 task_(Train, batch_size=ref.batch_size, progress=ref.progress, training=Ref('training'))
#             )
#             validate = (
#                 loads_(progress_bar, Ref('progress'), Ref('validation')) <<
#                 task_(Tester, batch_size=Ref('batch_size'), progress=Ref('progress'), validation=Ref('validation'))
#             )


# The major benefit is I can put this "trainer" tree in another
# module and just need to implement the functions

# class TrainerTree(Tree):

#     # if class is defined it will use "itself"

#     def __init__(self, train, validate, training_dataset, test_dataset, n_iterations):
#         pass

#     @task
#     class entry(Sequence):

#         load_datasets = action('load_datasets')
#         init_progress = action('init_progress')
#
#         @until
#         class T(Parallel):
    #         
    #         @task
    #         class train(Sequence):
    #             train = action('train', ref.learner, ref.dataset, store=store_ref())
    #             validate = action('validate')
    #             finished = cond('check_finished')
    #             update_progress = action('update_progress')
    #         @until
    #         check_stop = )
    # 

#     def reset(self):
#         pass

#     # tasks
#     def _load_datasets(self):
#         pass

#     def _train(self, learner, store):
#         pass

#     def _validate(self, learner, store):
#         pass

#     def _check_finished(self, store):
#         pass

#     def train(self, learner):

#         self.load_datasets()
#         self.initiate_progress()
#         while True:
#             self.update_progress()
#             self.train()
#             self.validate()
#             if self.check_finished():
#                 break


# class Trainer:

#     def __init__(self):
#         pass

#     def load_datasets(self):
#         pass

#     def train(self):
#         pass

#     def validate(self):
#         pass

# class Trainer:

#     def __init__(self, validator, trainer):
#        
#         # then you have to make sure that the arguments align
#         self._validator = validator
#         self._trainer = trainer

#     def _validate(self):
#         # need to write a validator for each trainer... or
#         # use composition
#         pass
    
#     def _train(self):
#         pass

#     def train(self):
        
#         for i in range(self._n_epochs):
#             self._train()
#             self._validate()
#     
#    behavior tree version is mor modular than this

