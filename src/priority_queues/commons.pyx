from timeit import default_timer

import numpy as np
import psutil

DTYPE = np.float64
DTYPE_PY = DTYPE
DTYPE_INF = <DTYPE_t>np.finfo(dtype=DTYPE).max
N_THREADS = psutil.cpu_count()

class Timer:
    def __init__(self):
        self._timer = default_timer
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start the timer."""
        self.start = self._timer()

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        self.interval = self.end - self.start