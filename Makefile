NUM_JOBS = 8
CXX      = g++
PRO      = gpt_demo

SPACE   ?= $(PWD)
LIBDIR  ?= $(SPACE)/lib
LIBSPEC ?= $(SPACE)/include
COMPILE ?= . $(SPACE)/source

CMAKE_CMD = mkdir -p build && cd build && cmake ..
CMAKE_MAKE = cd build

# FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER=$(CXX) -DABSL_INTERNAL_AT_LEAST_CXX20=OFF -DNDEBUG=0
FLAGS = 
DEBUG_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Debug
RELEASE_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Release

all : 
	@$(COMPILE) && $(CMAKE_CMD) $(DEBUG_FLAGS) && make -s -j$(NUM_JOBS)

build:
	@$(CMAKE_MAKE) && make -s -j$(NUM_JOBS)

debug_cuda:
	@cd debug/gpt && cuda-gdb -x ./cuda_gpt_demo.gdb

debug_test:
	@cd debug/test && gdb -x ./sampling_layer.gdb

# run :
# 	@cd build/bin && ./$(PRO) 

run :
	@cd build/bin && ./decoding_example 4 1 8 64 2048 30000 6 32 32 512 0 0.6 1

debug :
	cd debug/gpt && gdb -x ./gpt_demo.gdb

test_gemm:
	@cd build/bin && ./test_gemm

unittest :
	@cd build/bin && ./unittest

sampling_layer:
	@cd build/bin && ./sampling_layer

test_activation:
	@cd build/bin && ./test_activation

test_layernorm:
	@cd build/bin && ./test_layernorm 1 512 1
	
demo_tensor:
	@cd build/bin && ./demo_tensor 1 224 768 12 768 0

demo_vit:
	@cd build/bin && ./demo_vit

debug_vit:
	@cd debug && gdb -x ./demo_vit.gdb

gpt:
	@cd build/bin && ./gpt_example

debug_gpt:
	@cd debug/gpt && gdb -x ./gpt_debug.gdb

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python is not installed.  Aborting."; exit 1; }

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CRAL-C to abort]" response && rm -rf build/*

.PHONY: all run debug debug_cuda test clean check-python
