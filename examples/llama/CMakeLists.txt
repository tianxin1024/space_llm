
add_executable(llama_example llama_example.cc)
target_link_libraries(llama_example PUBLIC -lcublas -lcublasLt -lcudart )
