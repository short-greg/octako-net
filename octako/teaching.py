from abc import abstractmethod
import dataclasses
import typing
from sango.nodes import STORE_REF, Action, Status, Tree, Sequence, Parallel, action, cond, loads_, loads, neg, task, task_, until, var_, const_
from sango.vars import Const, Ref, Var, ref
from torch.types import Storage

from tqdm import tqdm
from functools import partial
from dataclasses import dataclass, is_dataclass
import pandas as pd
from torch.utils.data import  DataLoader


@dataclass
class Progress:

    name: str
    total_epochs: int
    cur_epoch: int = 0
    total_iterations: int = 0 
    cur_iteration: int = 0

    def to_dict(self):
        return dataclasses.asdict(self)


class ProgressRecorder(object):

    def __init__(self, default_epochs: int=1):
        self._progresses = {}
        self._cur_progress: str = None
        self._default_epochs = default_epochs
        self._completed = False
    
    def add(self, name: str, n_epochs: int=None, total_iterations: int=0, switch=True):
        if name in self._progresses:
            raise ValueError(f'Progress named {name} already exists.')
        
        n_epochs = n_epochs if n_epochs is not None else self._default_epochs

        self._progresses[name] = Progress(
            name, n_epochs, total_iterations=total_iterations
        )
        if switch: 
            self.switch(name)

    def switch(self, name: str):
        if name not in self._progresses:
            raise ValueError(f'Progress named {name} does not exist.')
        
        self._cur_progress = name
    
    def get(self, name: str):
        return self._progresses[name]
    
    @property
    def names(self):
        return set(self._progresses.keys())

    @property
    def cur(self) -> Progress:
        return self._progresses.get(self._cur_progress)
    
    def complete(self):
        self._completed = True
        
    @property
    def completed(self):
        return self._completed
    
    def adv_epoch(self, total_iterations=0):
        self.cur.cur_epoch += 1
        self.cur.total_iterations = total_iterations

    def adv_iter(self):
        self.cur.cur_iteration += 1


class Results:
    
    def __init__(self):
        
        self.df = pd.DataFrame()
        self._progress_cols = set()
        self._result_cols = set()
    
    def add_result(self, teacher: str, progress: Progress, results: typing.Dict[str, float]):

        self._progress_cols.update(
            set(progress.to_dict().keys())
        )

        self._result_cols.update(
            set(results.keys())
        )

        self.df = self.df.append({
            self.teacher_col: teacher,
            **progress.to_dict(),
            **results
        }, ignore_index=True)
    
    @property
    def teacher_col(self):
        return "Teacher"
    
    @property
    def result_cols(self):
        return set(self._result_cols)
    
    @property
    def progress_cols(self):
        return set(self._progress_cols)


class Teach(Action):
    
    result = const_()
    dataset = const_()
    progress = const_()
    batch_size = const_()
    learner = const_()

    def __init__(self, name: str):
        super().__init__(name)
        self._iter = None

    def _setup_progress(self):
        self._iter = iter(DataLoader(
            self.dataset.val, self.batch_size.val, shuffle=True
        ))
        n_iterations = len(self._iter)
        if self._name in self.progress.val.names:
            self.progress.val.switch(self._name)
            self.progress.val.adv_epoch(n_iterations)
        else:
            self.progress.val.add(
                self._name, total_iterations=n_iterations, switch=True
            )
    
    def _setup_iter(self):

        is_setup = self._iter is not None
        if not is_setup:
            self._setup_progress()
        else:
            self.progress.val.adv_iter()

    @abstractmethod
    def perform_action(self, x, t):
        pass

    def reset(self):
        super().reset()
        self._iter = None

    def act(self):
        self._setup_iter()

        try:
            x, t = next(self._iter)
        except StopIteration:
            self._iter = None
            return Status.SUCCESS
        
        result = self.perform_action(x, t)

        self.result.val.add_result(self._name, self.progress.val.cur, result)
        return Status.RUNNING


class Train(Teach):

    def perform_action(self, x, t):
        return self.learner.val.learn(x, t)


class Validate(Teach):
    
    def perform_action(self, x, t):
        return self.learner.val.test(x, t)


class Trainer(Tree):

    n_batches = var_(1)
    batch_size = var_(32)
    validation_dataset = var_()
    training_dataset = var_()
    learner = var_()
    results = var_()
    progress = var_()

    @task
    class entry(Parallel):
        update_progress_bar = action('update_progress_bar', STORE_REF)

        @task
        @until
        @neg
        class train(Sequence):
            to_finish = cond('to_finish', store=STORE_REF)
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

    def __init__(self, name: str):
        super().__init__(name)
        self._progress = ProgressRecorder()

    def load_datasets(self):
        pass

    def to_finish(self, store: Storage):
        cur_batch = store.get_or_add('cur_batch', 0)

        if cur_batch.val < self.n_batches.val:
            cur_batch.val += 1
            return Status.SUCCESS

        self._progress.complete()

        return Status.FAILURE

    def update_progress_bar(self, store: Storage):
        
        pbar = store.get_or_add('pbar', recursive=False)

        if self._progress.completed:
            if not pbar.is_empty(): 
                pbar.close()
                pbar.empty()
            return Status.SUCCESS
        
        if self._progress.cur is None:
            return Status.RUNNING
        if pbar.is_empty():
            pbar.val = tqdm(total=self._progress.cur.total_iterations)
        pbar.val.set_description_str(self._progress.cur.name)
        pbar.val.update(1)

        # self.pbar.total = lecture.n_lesson_iterations
        # self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
        # pbar.refresh()
        return Status.RUNNING

