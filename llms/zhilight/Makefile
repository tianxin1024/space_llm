NUM_JOBS = 8
CXX      = g++

PRO      = test_gemm

CMAKE_CMD = mkdir -p build && cd build && cmake ..

FLAGS = 
DEBUG_FLAGS = $(FLAGS) -DCMAKE_BUILD_TYPE:STRING=Debug


all :
	@$(CMAKE_CMD) $(DEBUG_FLAGS) && make -s -j$(NUM_JOBS) 

run :
	@cd build/tests && ./$(PRO)
