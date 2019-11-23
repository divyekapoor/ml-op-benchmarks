# Author: Divye Kapoor
# Date: Nov 22, 2019
#
# This file isn't that useful because ONNX Tracing is useless for the control
# flow used in FizzBuzz. It isn't able to handle the export.

import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from torch_fizz import TorchFizzBuzz

print("Saving ONNX model.")
mod = TorchFizzBuzz()
dummy = torch.tensor(100, dtype=torch.int32)
torch.onnx.export(mod, dummy, '/tmp/fizzbuzz.onnx', export_params=True, opset_version=10, do_constant_folding=False)

# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
print("Loading ONNX model.")
loaded_module = onnx.load('/tmp/fizzbuzz.onnx')
onnx.checker.check_model(loaded_module)
print(onnx.helper.printable_graph(loaded_module.graph))
print(dir(loaded_module))

print("Running ONNX model with onnxruntime.")
ort_session = ort.InferenceSession('/tmp/fizzbuzz.onnx')
start_ns = time.perf_counter_ns()
result = ort_session.run(None, {ort_session.get_inputs()[0].name: np.array([10000]).astype(np.int32)})
end_ns = time.perf_counter_ns()
print("Result: ", result)
print("Time (ONNX from Loaded) (ms): ", (end_ns - start_ns)/1e6)

print("Running ONNX model with small inputs (w/onnxruntime)")
start_ns = time.perf_counter_ns()
result = ort_session.run(None, {ort_session.get_inputs()[0].name: np.array([1000]).astype(np.int32)})
end_ns = time.perf_counter_ns()
print("Result: ", result)
print("Time (ONNX from Loaded) (ms): ", (end_ns - start_ns)/1e6)
