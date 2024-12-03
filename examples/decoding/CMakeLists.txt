
add_executable(decoding_example decoding_example.cc)
target_link_libraries(decoding_example PUBLIC -lcublas -lcublasLt -lcudart )
