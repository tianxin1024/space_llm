enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_CUDA_VISIBILITY_PRESET default)
set(TORCH_CUDA_ARCH_LIST "8.0+PTX;8.9")

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; \
		print(pybind11.get_cmake_dir())"
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

file(GLOB_RECURSE PYBIND_SOURCES "*.cpp")

pybind11_add_module(space ${PYBIND_SOURCES})
target_compile_definitions(space PUBLIC 
    VERSION_INFO=${EXAMPLE_VERSION_INFO})

target_link_libraries(space PRIVATE
    "bmengine"
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread")


