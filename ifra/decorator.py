import traceback
from copy import copy
from functools import wraps


def emit(_func):
    """Actor's methods should be decorated with that, to automatically send messages about the actor's current state."""
    def tags_decorator(func):
        @wraps(func)
        def wrapper_emit(*args, **kwargs):
            """Decorate a method of an object having an emitter to emit error when failing, and setting emitter's
            'doing' message to the function being executed."""
            if len(args) == 0:
                raise ValueError("Wrong use of 'emit_error' decorator. It must have at least one argument : self.")
            if not hasattr(args[0], "emitter"):
                raise ValueError("Wrong use of 'emit_error' decorator. First argument did not have the 'emitter'"
                                 "attribute.")
            self = args[0]
            emitter = self.emitter  # args[0] is 'self'
            try:
                doing = copy(emitter.doing)
                emitter.doing = func.__name__
                res = func(*args, **kwargs)
                emitter.doing = doing
                return res
            except (Exception, KeyboardInterrupt) as e:
                emitter.send(doing=None, error=traceback.format_exc())
                raise e
        return wrapper_emit

    return tags_decorator(_func)
