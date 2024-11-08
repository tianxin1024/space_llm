cmake_minimum_required(VERSION 3.10)

add_library(BaseSamplingLayer STATIC BaseSamplingLayer.cc)
set_property(TARGET BaseSamplingLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BaseSamplingLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BaseSamplingLayer PUBLIC -lcudart memory_utils)

