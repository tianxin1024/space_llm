cmake_minimum_required(VERSION 3.10)
project(tests)

enable_language(CXX)

add_executable(test_attention_rag_buffer test_attention_rag_buffer.cpp)
target_link_libraries(test_attention_rag_buffer LINK_PUBLIC bmengine)


