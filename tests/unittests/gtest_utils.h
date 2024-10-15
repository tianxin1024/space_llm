
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "utils/allocator.h"
#include "utils/cuda_utils.h"
#include "utils/memory_utils.h"
#include "utils/tensor.h"

namespace qk = space_llm;

namespace {

class QKTestBase : public testing::Test {
public:
    void SetUp() override {
        int device = 0;
        cudaGetDevice(&device);
        cudaStreamCreate(&stream);
        allocator = new qk::Allocator(device);
        allocator->setStream(stream);
    }

    void TearDown() override {
        // Automatically allocated CPU buffers should be released at the end of a test.
        // We don't need to care GPU buffers allocated by Allocator because they are
        // managed by the allocator.
        for (auto &buffer : allocated_cpu_buffers) {
            free(buffer);
        }
        allocated_cpu_buffers.clear();
        delete allocator;
        cudaStreamDestroy(stream);
    }

protected:
    cudaStream_t stream;
    qk::Allocator *allocator;
    std::vector<void *> allocated_cpu_buffers;
};

} // namespace
