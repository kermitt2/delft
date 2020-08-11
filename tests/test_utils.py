import logging
from functools import wraps

# derived from https://github.com/elifesciences/sciencebeam-trainer-delft/tree/develop/tests

LOGGER = logging.getLogger(__name__)


def log_on_exception(f: callable) -> callable:
    """
    Wraps function to log error on exception.
    That is useful for tests that log a lot of things,
    and pytest displaying the test failure at the top of the method.
    (there doesn't seem to be an option to change that)
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception('failed due to %s', repr(e))
            raise
    return wrapper
