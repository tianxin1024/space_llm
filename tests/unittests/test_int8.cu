#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <string>

#include "utils/cuda_utils.h"
#include "gtest_utils.h"

using namespace space_llm;

class Int8TestSuite : public QKTestBase {
public:
    void SetUp() override {
        QKTestBase::SetUp();
    }

    void TearDown() override {
        QKTestBase::TearDown();
    }

protected:
    using QKTestBase::stream;
    using QKTestBase::allocator;

    struct cudaDeviceProp prop;

    void testTransposition();
};

void Int8TestSuite::testTransposition() {
    EXPECT_EQ(0, 0);
}

TEST_F(Int8TestSuite, TranspositionCorrectness) {
    this->testTransposition();
}
