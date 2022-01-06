from sango.nodes import Action, Status, Task, TickDecorator, Tree, decorate, loads_, loads, task_, var
from sango.vars import Ref, Var
from torch.types import Storage
from octako.machinery.construction import Sequence
from octako.machinery.learners import Learner, Tester
from tqdm import tqdm
from functools import partial


def progress_bar(iterations: int):

    pbar: tqdm = None
    final_status = None

    def _(node: Task):
    
        nonlocal iterations
        if isinstance(iterations, Ref):
            iterations = iterations.shared(node._storage)

        def tick(node: Action, wrapped_tick):
            nonlocal pbar
            nonlocal final_status
            if pbar is None:
                pbar = tqdm(total=iterations)
                pbar.set_description_str(f'{node.name}')
                pbar.update(1)
                # TODO: How to get the results ub
                # self.pbar.total = lecture.n_lesson_iterations
                # self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
                pbar.refresh()
                return Status.RUNNING
            
            if final_status is None:

                status = wrapped_tick()
                if status.done:
                    final_status = status
                return Status.RUNNING

            pbar.close()
            return final_status
        
        return TickDecorator(node, tick)
    return _


def upto(iterations: int):

    def _(node: Task):
    
        i = 0
        nonlocal iterations
        if isinstance(iterations, Ref):
            iterations = iterations.shared(node._storage)

        def tick(node: Action, wrapped_tick):
            nonlocal i
            if i >= iterations:
                return Status.DONE
            result = wrapped_tick()
            if result == Status.FAILURE:
                return result
            
            i += 1
            if i == iterations:
                return Status.SUCCESS
            return Status.RUNNING
        
        return TickDecorator(node, tick)
    return _


def progress_bar_(iterations: int):

    return partial(progress_bar, iterations=iterations)


class Train(Action):
    
    dataset = Var()
    learner: Learner = Var()
    result = Var()

    def __init__(self, optim):

        self._optim = optim
        self._iter = None
        self._results = []

    def act(self):

        if self._iter is None:
            self._iter = iter(self.dataset)
        
        try:
            x, t = next(self._iter)
            self._results.append(self.learner.learn(x, t))
            return Status.RUNNING
        except StopIteration:
            pass
        
        yield Status.SUCCESS


class Test(Action):
    
    dataset = Var()
    tester: Tester = Var()
    result = Var()

    def __init__(self, optim):

        self._optim = optim
        self._results = []

    def reset(self):

        self._results = []

    def act(self):
        
        if self._iter is None:
            self._iter = iter(self.dataset)
        
        try:
            x, t = next(self._iter)
            self._results.append(self.tester.test(x, t))
            return Status.RUNNING
        except StopIteration:
            pass
        
        yield Status.SUCCESS


# def neg(node: Task):

#     def tick(node: Task, wrapped_tick):
#         status = wrapped_tick()
#         if status == Status.SUCCESS:
#             return Status.FAILURE
#         elif status == Status.FAILURE:
#             return Status.SUCCESS
#         return status
    
#     return TickDecorator(node, tick)

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


class Trainer(Tree):

    n_batches = var(10)
    batch_size = var(32)
    progress = var()
    material = var()

    class entry(Sequence):
        
        # variables
        training_dataset = var()
        validation_dataset = var()

        # tasks
        load_datasets = task_(LoadDatasets, Ref('material'))
        @upto(Ref('n_batches'))
        class train(Sequence):
            train = (
                loads_(progress_bar, Ref('progress'), Ref('training')) <<
                task_(Train, batch_size=Ref('batch_size'), progress=Ref('progress'), training=Ref('training'))
            )
            validate = (
                loads_(progress_bar, Ref('progress'), Ref('validation')) <<
                task_(Tester, batch_size=Ref('batch_size'), progress=Ref('progress'), validation=Ref('validation'))
            )


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

