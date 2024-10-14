#include <iostream>
#include <vector>
#include <unordered_map>

#include <gtest/gtest.h>

#include "utils/tensor.h"

using namespace space_llm;

namespace {

TEST(TensorMapTest, HasKeyCorrectness) {
    bool *v1 = new bool(true);
    float *v2 = new float[6]{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    Tensor t1 = Tensor{MEMORY_CPU, TYPE_BOOL, {1}, v1};
    Tensor t2 = Tensor{MEMORY_CPU, TYPE_FP32, {3, 2}, v2};

    TensorMap map({{"t1", t1}, {"t2", t2}});
    EXPECT_TRUE(map.isExist("t1"));
    EXPECT_TRUE(map.isExist("t2"));
    EXPECT_FALSE(map.isExist("t3"));

    delete v1;
    delete[] v2;
}
}; // namespace
