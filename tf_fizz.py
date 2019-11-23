import time
import tensorflow as tf


# https://github.com/tensorflow/tensorflow/issues/14132
# https://www.tensorflow.org/guide/saved_model
class FizzBuzz(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def model(self,
              n  # Shape [] -- int64 the max number to loop FizzBuzz to
              ):  # Returns counts for fizz, buzz and fizzbuzz. Shape: [1] with length 3
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


tfmod = FizzBuzz()
pymod = PyFizzBuzz()
COUNT = 100_000

# What's the code
print(tf.autograph.to_code(tfmod.model.python_function))

# Export the TensorFlow model and load it back
tf.saved_model.save(tfmod, '/tmp/fizzbuzz.m')
tf_loaded_model = tf.saved_model.load('/tmp/fizzbuzz.m')

perf_counter_ns_start = time.perf_counter_ns()
result = tfmod.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (TF Python) (ms): ', time_taken_ns / 1e6)

perf_counter_ns_start = time.perf_counter_ns()
result = tf_loaded_model.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (SavedModel) (ms): ', time_taken_ns / 1e6)

perf_counter_ns_start = time.perf_counter_ns()
result = pymod.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (Python3) (ms): ', time_taken_ns / 1e6)
