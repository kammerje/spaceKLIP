import logging
# Define logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

from contextlib import contextmanager

@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    param highest_level : the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable
    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""

    import sys, os

    prev_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")

    try:
        with open(os.devnull, "w") as fh:
            sys.stdout = fh
            yield
    finally:
        sys.stdout = prev_stdout # reset old stdout
