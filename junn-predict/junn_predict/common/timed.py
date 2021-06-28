"""Helper module to perform on the fly benchmarking/time keeping."""
import time


class Timed:
    """Context manager to keep track of elapsed time."""

    precision = 3

    def __init__(self, name=None):
        """
        Initialize the context manager, with an optional name.

        :param name: Name of the context to be time-tracked.
        """
        self.name = name
        self.elapsed = float('nan')

    def __enter__(self):
        """
        Call upon entry, records the start time.

        :return: self
        """
        self.time_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Call upon context finishing, will record the end time.

        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        self.time_end = time.time()

        self.elapsed = self.time_end - self.time_start

    def __float__(self):
        """
        Return the elapsed time as float.

        :return: Elapsed time as float.
        """
        return self.elapsed

    def __str__(self):
        """
        Return the elapsed time as string.

        :return: Elapsed time as string.
        """
        return ('%%.%df' % self.precision) % self.elapsed
