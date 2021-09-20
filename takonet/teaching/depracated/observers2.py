from .bulletins import Course, Entry
from .students import Student
from . import events
from .events import TeachingEvent
from .staff import Run, StandardTeacher
from .bulletins import RunProgress
import tqdm
import typing
import typing


class Observer(object):

    def __init__(
        self, name
    ):
        self._name = name

    @property
    def name(self):
        return self._name
        

class ProgressBar(Observer):

    def __init__(
        self, name: str, course: Course, listen_to: typing.List[StandardTeacher]
    ):
        """[Track progress of training]

        Args:
            name (str): [Name of progress bar]
            listen_to (typing.List[StandardTeacher]): [List of teacher to track progress for]
        """
        super().__init__(name)
        self.pbar = None
        self._course = course
        self._teachers = {}
        for teacher in listen_to:
            teacher.round_started_event.add_listener(self.enter)
            teacher.result_updated_event.add_listener(self.update)
            teacher.round_finished_event.add_listener(self.exit)
            self._teachers[teacher.name] = teacher

    @property
    def name(self):
        return self._name

    def enter(self, entry: Entry):
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm.tqdm(total=entry.progress.n_round_iterations)
        self.pbar.set_description_str(f'{entry.teacher_name} - {entry.section}')

    def update(self, entry: Entry):

        self.pbar.set_description_str(f'{entry.teacher_name} - {entry.section}')
        self.pbar.update(1)

        self.pbar.total = entry.progress.n_round_iterations
        self.pbar.set_postfix(entry.results.mean(axis=0).to_dict())
        self.pbar.refresh()

    def exit(self, entry: Entry):

        if self.pbar is not None:
            self.pbar.close()
        
        self.pbar = None


class TriggerCondition(object):
    """[Checks to see if the trigger should be executed.]
    """

    def check(self, progress: RunProgress):
        raise NotImplementedError


class IterationCondition(TriggerCondition):

    def __init__(self, period: int=1):
        assert period >= 1
        self._period = period

    def check(self, progress: RunProgress):
        return not bool((progress.cur_iteration + 1) % self._period)


class RoundCondition(object):

    def __init__(self, period: int=1):
        assert period >= 1
        self._period = period

    def check(self, progress: RunProgress):

        # TODO: epoch is not guaranteed to be here.. think how to handle this
        return not bool((progress.cur_round + 1) % self._period)


class RoundFinishedCondition(object):

    def check(self, progress: RunProgress):
        # TODO: epoch is not guaranteed to be here.. think how to handle this
        return progress.round_finished


class FinishedCondition(object):

    def check(self, progress: RunProgress):
        # TODO: epoch is not guaranteed to be here.. think how to handle this
        return progress.finished


class Trigger(Observer):

    def __init__(
        self, name: str, condition: TriggerCondition, 
        observing_event: TeachingEvent,
        observer_method: typing.Callable,
        course: Course
    ):
        """[summary]

        Args:
            name (str): [Name of the trigger.]
            condition (TriggerCondition): [Condition specifying when trigger should be enacted]
            observing_event (TeachingEvent): [Teaching event to observe]
            observer_method (typing.Callable): [Method to call]
        """
        self._name = name
        self._condition = condition
        self._observer_method = observer_method
        self._observing_event = observing_event
        self._observing_event.add_listener(self.on_trigger)
        self._course = course
    
    @property
    def name(self):
        return self._name
    
    def on_trigger(self, entry: Entry):
        progress = entry.progress
        cur_class = self._course.get_class(entry.class_id)
        student = cur_class.get_student(entry.student)
        if self._condition.check(progress):
            self._observer_method(student, "run")
            # run.add_response(response)


class TriggerBuilder(object):

    RESULT_UPDATED = 'update'
    ROUND_STARTED = 'round_update'
    ROUND_FINISHED = 'round_finished'
    STARTED = 'started'
    FINISHED = 'finished'

    def __init__(
        self, name: str, condition: TriggerCondition=None,
        observing_event=None,
    ):
        """[Set up and bulid a trigger]

        Args:
            name (str): [description]
            condition (TriggerCondition, optional): [description]. Defaults to None.
            observing_event ([type], optional): [description]. Defaults to None.
        """
        self._name = name
        self._condition = condition or FinishedCondition()
        self._observing_event = observing_event or self.ROUND_FINISHED
    
    def set_finished_condition(self):

        self._condition = FinishedCondition()
        return self
    
    def set_iteration_condition(self, period: int=1):

        self._condition = IterationCondition(period)
        return self
    
    def set_round_condition(self, period: int=1):

        self._condition = RoundCondition(period)
        return self
    
    def set_round_finished_condition(self):

        self._condition = RoundFinishedCondition()
        return self

    def set_observing_result_update(self):

        self._observing_event = self.RESULT_UPDATED
        return self
    
    def set_observing_round_started(self):

        self._observing_event = self.ROUND_STARTED
        return self

    def set_observing_round_finished(self):

        self._observing_event = self.ROUND_FINISHED
        return self

    def set_observing_started(self):
        self._observing_event = self.STARTED
        return self

    def set_observing_finished(self):

        self._observing_event = self.FINISHED
        return self
    
    def set_name(self, name: str):
        self._name = name
        return self

    def get_event(self, observing: StandardTeacher):
        """[Get the event that enacts the trigger from the teacher]

        Args:
            observing (StandardTeacher): [The teacher being observed]

        Returns:
            [type]: [description]
        """

        if self._observing_event ==  self.FINISHED:
            return observing.finished_event
        elif self._observing_event == self.RESULT_UPDATED:
            return observing.result_updated_event
        elif self._observing_event == self.ROUND_STARTED:
            return observing.round_started_event
        elif self._observing_event == self.ROUND_FINISHED:
            return observing.round_finished_event
        elif self._observing_event == self.STARTED:
            return observing.started_event

    def build(self, observing: Observer, observer_method: typing.Callable, course: Course):

        return Trigger(self._name, self._condition,  self.get_event(observing), observer_method, course)


class Audience(object):
    """[Convenience class for storing observers]
    """

    def __init__(self):

        self._members = {}
    
    def add(self, observer: Observer):

        self._members[observer.name] = observer
