#!/usr/bin/env bash

rm -rf cbuild
mkdir cbuild
cd cbuild
cmake ..
make
./torch_fizz
