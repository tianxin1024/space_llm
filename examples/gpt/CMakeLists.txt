
add_executable(gpt_example gpt_example.cc)
target_link_libraries(gpt_example PUBLIC -lcublas -lcublasLt -lcudart ParallelGpt)
