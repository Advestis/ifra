import pytest

from ifra.decorator import emit


class SimpleEmitter:
    def __init__(self):
        self.messages = {"doing": None}

    def __getattr__(self, item):
        if item == "messages":
            raise ValueError("NodeMessenger object's 'messages' not set")
        if item == "path":
            raise ValueError("NodeMessenger object's 'path' not set")
        if item not in self.messages:
            raise ValueError(f"No messages named '{item}' was found")
        return self.messages[item]

    def __setattr__(self, item, value):
        if item == "messages" or item == "path":
            super().__setattr__(item, value)
            return
        if item not in self.messages and item not in self.__class__.DEFAULT_MESSAGES:
            raise ValueError(f"Message named '{item}' is not allowed")
        self.messages[item] = value

    def send(self, **kwargs):
        for key in kwargs:
            self.messages[key] = kwargs[key]


class Actor:
    def __init__(self):
        self.emitter = SimpleEmitter()

    @emit
    def func_1(self):
        assert self.emitter.doing == "func_1"

        @emit
        def func_2(self_):
            assert self.emitter.doing == "func_2"
            return "chat"

        @emit
        def func_3():
            pass

        @emit
        def func_4(a):
            a += 1
            pass

        chat = func_2(self)
        assert chat == "chat"

        assert self.emitter.doing == "func_1"

        with pytest.raises(ValueError):
            func_3()

        with pytest.raises(ValueError):
            func_4(0)

        return "chien"

    @emit
    def failing(self):
        raise ValueError("coucou")


def test_decorator():
    a = Actor()
    chien = a.func_1()
    assert chien == "chien"
    assert a.emitter.doing is None
    with pytest.raises(ValueError) as e:
        a.failing()
        assert a.emitter.error is not None
        assert "coucou" in a.emitter.error
