cmake_minimum_required(VERSION 3.10)

project(zhilight)

enable_language(C)
enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(WITH_TESTING "Compile with unit testing" ON)

# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(tests)

if(WITH_TESTING)
	execute_process(
		COMMAND ${PYTHON_EXECUTABLE} -c "import torch; \
			print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
		OUTPUT_VARIABLE PYTORCH_CXX_ABI
		OUTPUT_STRIP_TRAILING_WHITESPACE
        # COMMAND_ECHO STDERR
        COMMAND_ERROR_IS_FATAL ANY)
    list(APPEND CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
    list(APPEND CMAKE_CUDA_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
    message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
endif()


execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; \
		print(pybind11.get_cmake_dir())"
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

file(GLOB_RECURSE PYBIND_SOURCES "src/py_export/*.cpp")

pybind11_add_module(C ${PYBIND_SOURCES})
target_compile_definitions(C PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


target_link_libraries(C PRIVATE
    # bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread"
)


if (WITH_TESTING)
    add_subdirectory(tests/py_export_internal)
    # add_subdirectory(src/nn/tests)
endif()
