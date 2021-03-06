import time


class Timed:

    precision = 3

    def __init__(self, name=None):
        self.name = name
        self.elapsed = float('nan')

    def __enter__(self):
        self.time_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_end = time.time()

        self.elapsed = self.time_end - self.time_start

    def __float__(self):
        return self.elapsed

    def __str__(self):
        return ('%%.%df' % self.precision) % self.elapsed
