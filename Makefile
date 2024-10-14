NUM_JOBS = $(shell nproc)
CXX      = g++
PRO      = demo

SPACE   ?= $(PWD)
LIBDIR  ?= $(SPACE)/lib
LIBSPEC ?= $(SPACE)/include
COMPILE ?= . $(SPACE)/source

CMAKE_CMD = mkdir -p build && cd build && cmake ..
CMAKE_MAKE = cd build

# FLAGS = -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_CXX_COMPILER=$(CXX) -DABSL_INTERNAL_AT_LEAST_CXX20=OFF
FLAGS = 
DEBUG_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Debug
RELEASE_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Release

all : 
	@$(COMPILE) && $(CMAKE_CMD) $(DEBUG_FLAGS) && make -s -j$(NUM_JOBS)

build:
	@$(CMAKE_MAKE) && make -s -j$(NUM_JOBS)

run :
	# @cd build/bin && ./$(PRO) 

debug :
	@cd build/bin && gdb -x ./init.gdb

unittest :
	@cd build/bin && ./unittest
	
demo_tensor:
	@cd build/bin && ./demo_tensor 1 224 768 12 768 0

check-python:
	@command -v python3 >/dev/null 2>&1 || { echo >&2 "Python is not installed.  Aborting."; exit 1; }

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CRAL-C to abort]" response && rm -rf build/*

.PHONY: all run test clean check-python
