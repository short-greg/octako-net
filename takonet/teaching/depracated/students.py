from takonet.machinery import learners


class Student(learners.Learner):

    def __init__(self, learner: learners.Learner, id: int, class_id: int):

        self._learner = learner
        self._id = id
        self._class_id = class_id

    def learn(self, x, t):
        """Function for learning the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        return self._learner.learn(x, t)

    def test(self, x, t):
        """Function for evaluating the mapping from x to t

        Args:
            x ([type]): The input values
            t ([type]): The target values to map to
        """
        return self._learner.test(x, t)
    
    @property
    def id(self):
        return self._id
    
    @property
    def class_id(self):
        return self._class_id

    def fields(self):
        return self._learner.fields

    @property
    def learner(self):
        return self._learner
