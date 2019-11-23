/*
Authored: dkapoor@pinterest.com
Date: Nov 5, 2019

How to compile and run:
$ g++ -std=c++17 -O3 fizz.cc -o fizz && ./fizz 100000

Provides a friendly error message if you miss something.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

using namespace std;

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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: fizz <count>\n eg. fizz 10000\n";
        return -1;
    }

    auto start = chrono::high_resolution_clock::now();
    cout << fizzbuzz(stoi(argv[1]));
    auto end = chrono::high_resolution_clock::now();
    cout << "Time taken (C++) (usec): "
        << chrono::duration_cast<std::chrono::microseconds>(end - start).count() << '\n';

    return 0;
}
