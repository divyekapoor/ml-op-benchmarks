# Dependencies: CMake, a recent clang / g++, Mac OSX
#

tfbench:
	./make_tfbench.sh

torchbench:
	./make_torchbench.sh

npbench:
	./make_npbench.sh

torch_native:
	./make_torch_native.sh

cc_native:
	./make_cc_native.sh

test: tfbench torchbench npbench torch_native cc_native

clean:
	git clean -fxd

