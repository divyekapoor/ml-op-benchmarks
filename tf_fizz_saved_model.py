import tensorflow as tf
import time

COUNT = 100_000
tf_loaded_model = tf.saved_model.load('/tmp/fizzbuzz.m')

perf_counter_ns_start = time.perf_counter_ns()
result = tf_loaded_model.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (SavedModel) (ms): ', time_taken_ns / 1e6)

perf_counter_ns_start = time.perf_counter_ns()
result = tf_loaded_model.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (SavedModel) (ms): ', time_taken_ns / 1e6)
