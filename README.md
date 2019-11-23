Tensorflow / PyTorch Op Benchmark
=================================

Goal
----
To test and benchmark Tensorflow and PyTorch op based computation 
against JIT'd (Python) and compiled (C++, PyTorch native) implementations.

Motivation
----------
We currently implement cross features using native C++ code outside of 
the Tensorflow Saved Model. The goal of this effort is to judge 
feasibility and performance impact of implementing these cross
features within the model file using op-based computation.

Doing so reduces a huge set of complexity in our training-serving systems
(because C++ doesn't necessarily have to be part of our training loop,
we can do with Python). However, without performance in the production
serving loop, this effort is blocked.

Outcomes
--------
Tensorflow Ops and PyTorch TorchScript are 2 orders of magnitude slower 
than similarly written native or Python code, making them useful in only 
very limited contexts. 

Benchmark code
--------------
```python
class FizzBuzz(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def model(self,
              n  # Shape [] -- int32 the max number to loop FizzBuzz to
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
```
Raw python: Running the same code without tf.Module and @tf.function.
Raw C++: Equivalent implementation in straight C++.

For PyTorch:
```python
class TorchFizzBuzz(torch.nn.Module):
    def __init__(self):
        super(TorchFizzBuzz, self).__init__()
        self.fizz = torch.tensor(0, requires_grad=False)
        self.buzz = torch.tensor(0, requires_grad=False)
        self.fizzbuzz = torch.tensor(0, requires_grad=False)

    def forward(self, n: torch.Tensor):
        i = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        self.fizz = torch.zeros(1)
        self.buzz = torch.zeros(1)
        self.fizzbuzz = torch.zeros(1)
        while i < n:
            if i % 6 == 0:
                self.fizzbuzz += 1
            elif i % 3 == 0:
                self.buzz += 1
            elif i % 2 == 0:
                self.fizz += 1
            i += 1
        return torch.stack([self.fizz, self.buzz, self.fizzbuzz])
```

Benchmark System
----------------

```
Mac OS-X Mojave: 10.14.6 (18G1012)
MacBook Pro (13-inch, 2018, Four Thunderbolt 3 Ports)
2.3 GHz Intel Core i5

Hardware Overview:

  Model Name:	MacBook Pro
  Model Identifier:	MacBookPro15,2
  Processor Name:	Intel Core i5
  Processor Speed:	2.3 GHz
  Number of Processors:	1
  Total Number of Cores:	4
  L2 Cache (per Core):	256 KB
  L3 Cache:	6 MB
  Hyper-Threading Technology:	Enabled
  Memory:	16 GB
  Boot ROM Version:	1037.40.124.0.0 (iBridge: 17.16.11081.0.0,0)
```

Performance Tables
------------------


| FizzBuzz Iteration Counts     | 100000              |                          |                    |                |
| -------------------------     | -----------------   | -----------------------  | ------------------ | -------------  |
|                               | Method Latency (ms) | Iteration Latency (usec) | Python Multiplier  | C++ Multiplier |
| Tensorflow Python             | 4087                | 40.87                    | **227.06**         | 24327          |
| Tensorflow Saved Model Python | 4046                | 40.46                    | **224.78**         | 24083          |
| Raw Python                    | 18                  | 0.18                     | 1.00               | 107            |
| Raw C++                       | 0.168               | 0.00168                  | 0.01               | 1              |


| FizzBuzz Iteration Counts                            | 100000            |                         |                    |                |
| -------------------------                            | ----------------- | ----------------------- | ------------------ | -------------  |
|                                                      | Raw Latency (ms)  | Per Run Latency (usec)  | Python Multiplier  | C++ Multiplier |
| PyTorch Python                                       | 4007              | 40.07                   | 222.61             | 23851          |
| PyTorch TorchScript Python (from Loaded TorchScript) | 2830              | 28.3                    | **157.22**         | 16845          |
| PyTorch TorchScript C++ (Native)                     | 255               | 2.55                    | **14.17**          | 1518           |
| PyTorch TorchScript C++ (Native + ATen Tensors)      | 252               | 2.52                    | **14.00**          | 1500           |
| Raw Python                                           | 18                | 0.18                    | 1.00               | 107            |
| Raw C++                                              | 0.168             | 0.00168                 | 0.01               | 1              |

Expected Performance
--------------------

Both systems should be performing close to the speed of Raw Python (if not faster).
In practice, the systems are performing much slower than expected.


Issues Filed
------------

Tensorflow: https://github.com/tensorflow/tensorflow/issues/34500

PyTorch: TBD

How To Reproduce
----------------

```bash
$ make tfbench
```

```bash
$ make torchbench
```

```bash
$ make torch_native
```

```bash
$ make cc_native
```
