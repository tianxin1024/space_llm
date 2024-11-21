
# add_executable(gpt_example gpt_example.cc)
# target_link_libraries(gpt_example PUBLIC -lcublas -lcublasLt -lcudart ParallelGpt)


add_executable(gpt_demo gpt_demo.cc)
target_link_libraries(gpt_demo PUBLIC -lcublas -lcublasLt -lcudart Gpt)
