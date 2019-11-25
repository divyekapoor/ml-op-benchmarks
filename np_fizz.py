import numpy as np
import time


# Uses Array Ops throughout.
# Makes it roughly comparable to other "array-based" processing.
class NumPyFizzBuzz:
    def model(self, n):
        fizz = np.array(0)
        buzz = np.array(0)
        fizzbuzz = np.array(0)
        # Force everything to be a numpy single element array, for an even comparison
        for i in np.arange(n)[:, np.newaxis]:
            if i % 6 == 0:
                fizzbuzz += 1
            elif i % 3 == 0:
                buzz += 1
            elif i % 2 == 0:
                fizz += 1
        return [fizz, buzz, fizzbuzz]


class PyFizzBuzz:
    def model(self, n):
        fizz = 0
        buzz = 0
        fizzbuzz = 0
        for i in range(n):
            if i % 6 == 0:
                fizzbuzz += 1
            elif i % 3 == 0:
                buzz += 1
            elif i % 2 == 0:
                fizz += 1
        return [fizz, buzz, fizzbuzz]


numpymod = NumPyFizzBuzz()
pymod = PyFizzBuzz()
COUNT = 100_000

perf_counter_ns_start = time.perf_counter_ns()
result = numpymod.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (NumPy Python) (ms): ', time_taken_ns / 1e6)

perf_counter_ns_start = time.perf_counter_ns()
result = pymod.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (Raw Python3) (ms): ', time_taken_ns / 1e6)
