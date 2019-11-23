/**
 * PyTorch FizzBuzz (Native code)
 *
 * This file implements FizzBuzz with native PyTorch Tensor code.
 * There are 2 implementations: one using regular torch::Tensor and the other using at::Tensor
 * These are referred to as PyTorch Native and PyTorch ATNative implementations.
 *
 * Please see the README.md for details.
 */
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <torch/csrc/jit/import.h>

using namespace std;

struct FizzBuzz : public torch::nn::Module {
    FizzBuzz() {
    }

    torch::Tensor forward(torch::Tensor n) {
        int nmax = n.item().toInt();
        torch::Tensor fizz = torch::zeros(1);
        torch::Tensor buzz = torch::zeros(1);
        torch::Tensor fizzbuzz = torch::zeros(1);
        for (int i = 0; i < nmax; ++i) {
            if (i % 6 == 0) {
                fizzbuzz += 1;
            } else if (i % 3 == 0) {
                buzz += 1;
            } else if (i % 2 == 0) {
                fizz += 1;
            }
        }
        return torch::stack({fizz, buzz, fizzbuzz});
    }
};

struct FizzBuzzAT : public torch::nn::Module {
    FizzBuzzAT() {
    }

    torch::Tensor forward(at::Tensor n) {
        int nmax = n.item().toInt();
        at::Tensor fizz = at::zeros(1);
        at::Tensor buzz = at::zeros(1);
        at::Tensor fizzbuzz = at::zeros(1);
        for (int i = 0; i < nmax; ++i) {
            if (i % 6 == 0) {
                fizzbuzz += 1;
            } else if (i % 3 == 0) {
                buzz += 1;
            } else if (i % 2 == 0) {
                fizz += 1;
            }
        }
        return torch::stack({fizz, buzz, fizzbuzz});
    }
};

// C++ template to print vector container elements
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]\n";
    return os;
}

vector<int> fizzbuzz(int n) {
    int fizz = 0, buzz = 0, fizzbuzz = 0;
    for (int i = 0; i < n; ++i) {
        if (i % 6 == 0) ++fizzbuzz;
        else if (i % 3 == 0) ++buzz;
        else if (i % 2 == 0) ++fizz;
    }
    return {fizz, buzz, fizzbuzz};
}

vector<int> fizzbuzz_volatile(volatile int n) {
    volatile int fizz = 0, buzz = 0, fizzbuzz = 0;
    for (int i = 0; i < n; ++i) {
        if (i % 6 == 0) ++fizzbuzz;
        else if (i % 3 == 0) ++buzz;
        else if (i % 2 == 0) ++fizz;
    }
    return {fizz, buzz, fizzbuzz};
}

int main(int argc, char* argv[]) {
    torch::NoGradGuard no_grad_guard;
    cout << "Loading model.\n";
    auto start = chrono::high_resolution_clock::now();
    torch::jit::script::Module model = torch::jit::load("/tmp/fizzbuzz.pyt");
    model.eval();
    auto end = chrono::high_resolution_clock::now();
    cout << "Time taken (PyTorch Load) (ms): "
        << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

    start = chrono::high_resolution_clock::now();
    FizzBuzz cpp_model;
    cpp_model.eval();
    end = chrono::high_resolution_clock::now();
    cout << "Time taken (PyTorch Native Model Build) (ms): "
        << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

    start = chrono::high_resolution_clock::now();
    FizzBuzzAT at_cpp_model;
    at_cpp_model.eval();
    end = chrono::high_resolution_clock::now();
    cout << "Time taken (PyTorch Native Model Build) (ms): "
        << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';


    for (int i = 0; i < 3; ++i) {
        cout << "\n\nRun: " << i << "\n";
        cout << "=========================================\n";
        auto n = torch::tensor(100000);
        start = chrono::high_resolution_clock::now();
        auto result = model.forward({n});
        end = chrono::high_resolution_clock::now();
        cout << "Result: [" << result << "]\n";
        cout << "Time taken (PyTorch Loaded) (ms): "
            << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

        start = chrono::high_resolution_clock::now();
        result = cpp_model.forward({n});
        end = chrono::high_resolution_clock::now();
        cout << "Result: [" << result << "]\n";
        cout << "Time taken (PyTorch Native) (ms): "
            << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

        start = chrono::high_resolution_clock::now();
        result = at_cpp_model.forward({n});
        end = chrono::high_resolution_clock::now();
        cout << "Result: [" << result << "]\n";
        cout << "Time taken (PyTorch ATNative) (ms): "
            << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';

        start = chrono::high_resolution_clock::now();
        auto vresult = fizzbuzz(100000);
        end = chrono::high_resolution_clock::now();
        cout << "Result: " << vresult;
        cout << "Time taken (C++ Native Run) (ms): "
            << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';
        cout << "Time taken (C++ Native Run) (usec): "
            << chrono::duration_cast<std::chrono::microseconds>(end - start).count() << '\n';

        start = chrono::high_resolution_clock::now();
        vresult = fizzbuzz_volatile(100000);
        end = chrono::high_resolution_clock::now();
        cout << "Result: " << vresult;
        cout << "Time taken (C++ Native Volatile) (ms): "
            << chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << '\n';
        cout << "Time taken (C++ Native Volatile) (usec): "
            << chrono::duration_cast<std::chrono::microseconds>(end - start).count() << '\n';
    }
    return 0;
}
