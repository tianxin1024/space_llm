cmake_minimum_required(VERSION 3.10)

add_library(BaseSamplingLayer STATIC BaseSamplingLayer.cc)
set_property(TARGET BaseSamplingLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET BaseSamplingLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(BaseSamplingLayer PUBLIC -lcudart sampling_penalty_kernels memory_utils)

add_library(TopKSamplingLayer STATIC TopKSamplingLayer.cu)
set_property(TARGET TopKSamplingLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TopKSamplingLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TopKSamplingLayer PUBLIC -lcudart BaseSamplingLayer sampling_topk_kernels)

add_library(TopPSamplingLayer STATIC TopPSamplingLayer.cu)
set_property(TARGET TopPSamplingLayer PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET TopPSamplingLayer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(TopPSamplingLayer PUBLIC -lcudart BaseSamplingLayer sampling_topp_kernels)

