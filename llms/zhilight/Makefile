NUM_JOBS = 8
CXX      = g++

PRO      = test_attention_rag_buffer

CMAKE_CMD = mkdir -p build && cd build && cmake ..

FLAGS = 
DEBUG_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Debug

all :
	@$(CMAKE_CMD) $(DEBUG_FLAGS) && make -s -j$(NUM_JOBS) 

run :
	# @cd build/tests && ./$(PRO)
	@cd build/src/nn/tests && ./$(PRO)

debug:
	@echo $(shell ./debug/create_gdb $(PRO)) && gdb -x ./debug/$(PRO).gdb

bug :
	@gdb -x ./debug/debug_softmax.gdb

clean:
	read -r -p "This will delete the contents of build/*. Are you sure? [CRAL-C to abort]" response && rm -rf build/*

.PHONY: all run debug clean
