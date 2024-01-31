from contextlib import redirect_stdout
import os
from io import StringIO
import warnings
import functools


def lazy_evaluation_with_caching(method):
    cache = {}

    def wrapper(self):

        instance_cache = cache.setdefault(id(self), {'version': None, 'result': None})
        if instance_cache['version'] == self.grid_version:
            # print("in wrapper if")

            return instance_cache['result']
        else:
            # print("in wrapper else")
            result = method(self)
            instance_cache['version'] = self.grid_version
            instance_cache['result'] = result
            return result

    return wrapper

# def suppress_print(func):
#     def wrapper(*args, **kwargs):
#         with open(os.devnull, 'w') as devnull:
#             with redirect_stdout(devnull):
#                 return func(*args, **kwargs)
#     return wrapper

def suppress_warnings(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Suppress specific warnings, in this case, RuntimeWarnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return func(*args, **kwargs)
    return wrapper


def suppress_print(func):
    def wrapper(*args, **kwargs):
        # Create a string buffer to capture output
        with StringIO() as buf:
            with redirect_stdout(buf):
                return func(*args, **kwargs)
    return wrapper
