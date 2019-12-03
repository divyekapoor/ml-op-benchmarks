import time

import torch

class TorchFizzBuzzVec(torch.nn.Module):
    def __init__(self):
        super(TorchFizzBuzzVec, self).__init__()

    def forward(self, n: torch.Tensor):
        x = torch.arange(n)
        ones = torch.ones(n)
        zeros = torch.zeros(n)

        fizzbuzz = torch.sum(torch.where(x % 6 == 0, ones, zeros)).unsqueeze(0)
        buzz = torch.sum(torch.where((x % 3 == 0) & (x % 6 != 0), ones, zeros)).unsqueeze(0)
        fizz = torch.sum(torch.where((x % 2 == 0) & (x % 6 != 0), ones, zeros)).unsqueeze(0)

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

print("Saving PyTorch vectorized model.")
mod = TorchFizzBuzzVec()
# Convert to code
jit_script = torch.jit.script(mod)
print(jit_script)
print(jit_script.code)
torch.jit.save(jit_script, '/tmp/fizzbuzz.pyt')

mod = TorchFizzBuzzVec()
mod.eval()
start_ns = time.perf_counter_ns()
result = mod.forward(n)
end_ns = time.perf_counter_ns()
print("Result: ", result)
print("Time (PyTorch vectorized) (ms): ", (end_ns - start_ns)/1e6)

mod = TorchFizzBuzzVec()
mod.eval()
with torch.jit.optimized_execution(True):
    start_ns = time.perf_counter_ns()
    result = mod.forward(n)
    end_ns = time.perf_counter_ns()

print("Result: ", result)
print("Time (PyTorch vectorized, optimized=True) (ms): ", (end_ns - start_ns)/1e6)

mod = TorchFizzBuzzVec()
mod.eval()
with torch.jit.optimized_execution(False):
    start_ns = time.perf_counter_ns()
    result = mod.forward(n)
    end_ns = time.perf_counter_ns()
print("Result: ", result)
print("Time (PyTorch vectorized, optimized=False) (ms): ", (end_ns - start_ns)/1e6)

print("Loading PyTorch model.")
loaded_module = torch.jit.load('/tmp/fizzbuzz.pyt')
loaded_module.eval()
start_ns = time.perf_counter_ns()
result = loaded_module.forward(n)
end_ns = time.perf_counter_ns()
print("Result: ", result)
print("Time (PyTorch vectorized from Loaded) (ms): ", (end_ns - start_ns)/1e6)

pymod = PyFizzBuzz()
perf_counter_ns_start = time.perf_counter_ns()
result = pymod.model(COUNT)
perf_counter_ns_end = time.perf_counter_ns()
time_taken_ns = perf_counter_ns_end - perf_counter_ns_start
print('Result: ', result)
print('Time taken (Python3) (ms): ', time_taken_ns / 1e6)
