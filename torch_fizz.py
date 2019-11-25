import time

import torch

class TorchFizzBuzz(torch.nn.Module):
    def __init__(self):
        super(TorchFizzBuzz, self).__init__()

    def forward(self, n: torch.Tensor):
        i = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        fizz = torch.zeros(1)
        buzz = torch.zeros(1)
        fizzbuzz = torch.zeros(1)
        while i < n:
            if i % 6 == 0:
                fizzbuzz += 1
            elif i % 3 == 0:
                buzz += 1
            elif i % 2 == 0:
                fizz += 1
            i += 1
        return torch.stack([fizz, buzz, fizzbuzz])


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


torch.no_grad()
COUNT = 100000
n = torch.tensor(COUNT, dtype=torch.int32)

print("Saving PyTorch model.")
mod = TorchFizzBuzz()
# Convert to code
jit_script = torch.jit.script(mod)
print(jit_script)
print(jit_script.code)
torch.jit.save(jit_script, '/tmp/fizzbuzz.pyt')

mod = TorchFizzBuzz()
mod.eval()
start_ns = time.perf_counter_ns()
result = mod.forward(n)
end_ns = time.perf_counter_ns()
print("Result: ", result)
print("Time (PyTorch) (ms): ", (end_ns - start_ns)/1e6)

mod = TorchFizzBuzz()
mod.eval()
with torch.jit.optimized_execution(True):
    start_ns = time.perf_counter_ns()
    result = mod.forward(n)
    end_ns = time.perf_counter_ns()

print("Result: ", result)
print("Time (PyTorch optimized=True) (ms): ", (end_ns - start_ns)/1e6)


mod = TorchFizzBuzz()
mod.eval()
with torch.jit.optimized_execution(False):
    start_ns = time.perf_counter_ns()
    result = mod.forward(n)
    end_ns = time.perf_counter_ns()
print("Result: ", result)
print("Time (PyTorch optimized=False) (ms): ", (end_ns - start_ns)/1e6)

print("Loading PyTorch model.")
loaded_module = torch.jit.load('/tmp/fizzbuzz.pyt')
loaded_module.eval()
start_ns = time.perf_counter_ns()
result = loaded_module.forward(n)
end_ns = time.perf_counter_ns()
print("Result: ", result)
print("Time (PyTorch from Loaded) (ms): ", (end_ns - start_ns)/1e6)

pymod = PyFizzBuzz()
perf_counter_ns_start = time.perf_counter_ns()
result = pymod.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (Python3) (ms): ', time_taken_ns / 1e6)
