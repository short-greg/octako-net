from takonet.teaching.dojos import Lecture, Observer, Course
import typing
import tqdm


class ProgressBar(Observer):

    def __init__(
        self, name: str, course: Course, listen_to: typing.List[str]
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
        self._listen_to = set(listen_to)
        self._name = name

        course.lesson_started_event.add_listener(self.enter)
        course.result_updated_event.add_listener(self.update)
        course.lesson_finished_event.add_listener(self.exit)

    @property
    def name(self):
        return self._name

    def enter(self, name: str):
        if name not in self._listen_to:
            return
        
        lecture: Lecture = self._course.get_lecture(name)

        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm.tqdm(total=lecture.n_lesson_iterations)
        self.pbar.set_description_str(f'{name} - {lecture.cur_lesson_iteration}')

    def update(self, name: str):
        if name not in self._listen_to:
            return
        
        lecture: Lecture = self._course.get_lecture(name)

        self.pbar.set_description_str(f'{name} - {lecture.cur_lesson_iteration}')
        self.pbar.update(1)

        self.pbar.total = lecture.n_lesson_iterations
        self.pbar.set_postfix(lecture.results.mean(axis=0).to_dict())
        self.pbar.refresh()

    def exit(self, name: str):
        if name not in self._listen_to:
            return
        
        lecture: Lecture = self._course.get_lecture(name)

        if self.pbar is not None:
            self.pbar.close()
        
        self.pbar = None


class TriggerCondition(object):
    """[Checks to see if the trigger should be executed.]
    """

    def check(self, lecture: Lecture):
        raise NotImplementedError


class IterationCondition(TriggerCondition):

    def __init__(self, period: int=1):
        assert period >= 1
        self._period = period

    def check(self, lecture: Lecture):
        return not bool((lecture.cur_iteration) % self._period)


class LessonCondition(object):

    def __init__(self, period: int=1):
        assert period >= 1
        self._period = period

    def check(self, lecture: Lecture):
        # TODO: epoch is not guaranteed to be here.. think how to handle this
        return not bool((lecture.cur_lesson) % self._period)


class LessonFinishedCondition(object):

    def check(self, lecture: Lecture):
        # TODO: epoch is not guaranteed to be here.. think how to handle this
        return lecture.lesson_finished


class FinishedCondition(object):

    def check(self, lecture: Lecture):
        # TODO: epoch is not guaranteed to be here.. think how to handle this
        return lecture.finished


class Trigger(Observer):

    def __init__(
        self, name: str, condition: TriggerCondition, 
        observer_method: typing.Callable,
        course: Course, listen_to_event: str, 
    ):
        """[summary]

        Args:
            name (str): [Name of the trigger.]
            condition (TriggerCondition): [Condition specifying when trigger should be enacted]
            observing_event (TeachingEvent): [Teaching event to observe]
            observer_method (typing.Callable): [Method to call]
        """
        self._name = name

        self._course = course
        self._course.listen_to(listen_to_event, self.on_trigger)
        self._condition = condition
        self._observer_method = observer_method
        self._course = course
    
    @property
    def name(self):
        return self._name
    
    def on_trigger(self, name: str):
        lecture = self._course.get_cur_lecture(name)
        if self._condition.check(lecture):
            self._observer_method()


class TriggerInviter(object):

    RESULT_UPDATED = 'result_updated'
    LESSON_STARTED = 'lesson_started'
    LESSON_FINISHED = 'lesson_finished'
    STARTED = 'started'
    FINISHED = 'finished'
    ADVANCED = 'advanced'

    def __init__(
        self, name: str, callback: typing.Callable[[], typing.NoReturn], condition: TriggerCondition=None,
        observing_event=None,
    ):
        """[Set up and bulid a trigger]

        Args:
            name (str): [description]
            condition (TriggerCondition, optional): [description]. Defaults to None.
            observing_event ([type], optional): [description]. Defaults to None.
        """
        self._name = name
        self._condition = condition or LessonFinishedCondition()
        self._observing_event = observing_event or self.LESSON_FINISHED
        self._callback = callback

    def set_finished_condition(self):

        self._condition = FinishedCondition()
        return self
    
    def set_iteration_condition(self, period: int=1):

        self._condition = IterationCondition(period)
        return self
    
    def set_lesson_condition(self, period: int=1):

        self._condition = LessonCondition(period)
        return self
    
    def set_round_finished_condition(self):

        self._condition = LessonFinishedCondition()
        return self

    def set_observing_result_update(self):

        self._observing_event = self.RESULT_UPDATED
        return self
    
    def set_observing_lesson_started(self):

        self._observing_event = self.LESSON_STARTED
        return self

    def set_observing_lesson_finished(self):

        self._observing_event = self.LESSON_FINISHED
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
    
    def invite(self, course: Course):

        return Trigger(self._name, self._callback, self._condition, self._observing_event, course)
