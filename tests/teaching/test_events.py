from takonet.teaching import events


class Listener(object):

    def __init__(self):
        self.called = False

    def call(self, name):
        self.called = True


class LateListener(object):

    def __init__(self, listener: Listener):
        self.listener = listener
        self.called_after = None

    def call(self, name):
        if self.listener.called:
            self.called_after = True
        else:
            self.called_after = False


class TestTeachingEvent(object):


    def test_invoke_listener(self):

        listener = Listener()
        event = events.TeachingEvent[str]()
        event.add_listener(listener.call)
        event.invoke("X")
        assert listener.called is True
    
    def test_invoke_listener_removed(self):

        listener = Listener()
        event = events.TeachingEvent[str]()
        event.add_listener(listener.call)
        event.remove_listener(listener.call)
        event.invoke("X")
        assert listener.called is False

    def test_invoke_late_listener_base(self):

        listener1 = Listener()
        listener = LateListener(listener1)
        event = events.TeachingEvent[str]()
        event.add_late_listener(listener.call)
        event.invoke("X")
        assert listener.called_after is False


    def test_invoke_late_listener_after(self):

        listener1 = Listener()
        listener = LateListener(listener1)
        event = events.TeachingEvent[str]()
        event.add_listener(listener1.call)
        event.add_late_listener(listener.call)
        event.invoke("X")
        assert listener.called_after is True

    def test_invoke_late_listener_removed(self):

        listener1 = Listener()
        listener = LateListener(listener1)
        event = events.TeachingEvent[str]()
        event.add_late_listener(listener.call)
        event.remove_late_listener(listener.call)
        event.invoke("X")
        assert listener.called_after is None
