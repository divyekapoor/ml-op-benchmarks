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

Performance Tables
------------------

Tensorflow:

| FizzBuzz Iteration Counts      | 100000              |                          |                    |                |
| -------------------------      | -----------------   | -----------------------  | ------------------ | -------------  |
|                                | Method Latency (ms) | Iteration Latency (usec) | Python Multiplier  | C++ Multiplier |
| Tensorflow Python              | 4087                | 40.87                    | **227.06**         | 24327          |
| Tensorflow Saved Model Python  | 4046                | 40.46                    | **224.78**         | 24083          |
| Tensorflow Python no Autograph | 3981                | 39.81                    | **221.16**         | 23696          |
| NumPy Python                   | 420                 | 4.2                      | **23.3**           | 2500           |
| **Tensorflow Python with XLA** | **81**              | **0.81**                 | **4.5**            | 482            |
| Raw Python                     | 18                  | 0.18                     | 1.00               | 107            |
| Raw C++                        | 0.168               | 0.00168                  | 0.01               | 1              |

PyTorch:

| FizzBuzz Iteration Counts                            | 100000              |                          |                    |                |
| -------------------------                            | -----------------   | -----------------------  | ------------------ | -------------  |
|                                                      | Method Latency (ms) | Iteration Latency (usec) | Python Multiplier  | C++ Multiplier |
| PyTorch Python                                       | 4007                | 40.07                    | 222.61             | 23851          |
| PyTorch TorchScript Python (from Loaded TorchScript) | 2830                | 28.3                     | **157.22**         | 16845          |
| PyTorch TorchScript C++ (Native)                     | 255                 | 2.55                     | **14.17**          | 1518           |
| PyTorch TorchScript C++ (Native + ATen Tensors)      | 252                 | 2.52                     | **14.00**          | 1500           |
| NumPy Python                                         | 420                 | 4.2                     | **23.3**           | 2500           |
| Raw Python                                           | 18                  | 0.18                     | 1.00               | 107            |
| **PyTorch Vectorized**                               | **7.8**             | **0.078**                | **0.43**           | 46             |
| **PyTorch Vectorized (optimized=True)**              | **5.03**            | **0.050**                | **0.28**           | 29             |
| **PyTorch Vectorized (optimized=False)**             | **4.74**            | **0.047**                | **0.26**           | 28             |
| Raw C++                                              | 0.168               | 0.00168                  | 0.01               | 1              |


**Note**: These numbers are meant to be indicative. We're not looking at small
micro-differences of O(10%), the differences are in orders of magnitude. The
benchmarked numbers are stable to within 10% on repeated runs, so feel free to
assume equivalency for similarly placed values. However, note that the
error bars are not large enough to explain the typical 10000% differences seen above.
These differences are indicative of inefficiencies in software.

Expected Performance
--------------------

Both systems should be performing close to the speed of Raw Python (if not faster).
Even allowing for significant overhead, they should be within 10x of Raw Python.
Looping over 100K values should not take 4 seconds.
In practice, the systems are performing much slower than expected.


Issues Filed
------------

Tensorflow: https://github.com/tensorflow/tensorflow/issues/34500

PyTorch: https://github.com/pytorch/pytorch/issues/30365


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
        return torch.stack([self.fizz, self.buzz, self.fizzbuzz])
```

For NumPy:
```python
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

```
Tensorflow: 2.1.0-dev20200102 (tf-nightly, similar results with tf-2-stable)
PyTorch: 1.3.0-mac
```

All runs on CPU.

How To Reproduce
----------------

```bash
$ make tfbench
```

```bash
$ make torchbench
```

```bash
$ make npbench
```

```bash
$ make torch_native
```

```bash
$ make cc_native
```
