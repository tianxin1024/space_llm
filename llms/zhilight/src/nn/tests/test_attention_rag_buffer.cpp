
#include "core/core.h"
#include "functions/typecast.h"
#include "logger/kernel_time_trace.hpp"
#include "nn/attention/attention_kernel.h"
#include "logger/std_log_op.hpp"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>
#include <cuda.h>
#include <stdlib.h>

using namespace bmengine::core;
using namespace bmengine::functions;
using namespace bmengine;
using std::vector;

Tensor random_float(const Context &ctx, const vector<size_t> &shape, DataType dtype = DataType::kHalf) {
    Tensor tensor = ctx.tensor(shape, DataType::kFloat);
    vector<float> data(tensor.numel(), 0.f);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = float((rand() % 1000)) / 10.;
    }
    tensor.from_buffer(data.data());
    return typecast(ctx, tensor, dtype);
}

void test_mq_rag_buffer1(Context &ctx, size_t batch, size_t len_q, vector<int> lens,
                         size_t num_kv_heads, size_t m_query, size_t dim_head, int algo_id, int round) {
    float scale = 0.001f;
    srand(327U);

    Tensor q = random_float(ctx, {batch, len_q, num_kv_heads * m_query, dim_head});

    vector<Tensor> key_buffers(batch);
    vector<Tensor> val_buffers(batch);
    vector<void *> key_buf_addresses(batch);
    vector<void *> val_buf_addresses(batch);
    while (lens.size() < batch) {
        lens.push_back(lens[0]);
    }
    lens.resize(batch);
    int sum_len = std::reduce(lens.begin(), lens.end());
    Tensor mask = ctx.tensor_of(vector<int8_t>(batch * sum_len, 1));
    Tensor buf_lens = ctx.tensor_of(lens);

    for (size_t i = 0; i < batch; ++i) {
        key_buffers[i] = random_float(ctx, {num_kv_heads, size_t(lens[i]), dim_head});
        val_buffers[i] = random_float(ctx, {num_kv_heads, size_t(lens[i]), dim_head});
        key_buf_addresses[i] = key_buffers[i].data();
        val_buf_addresses[i] = val_buffers[i].data();
    }
    Tensor key_buf_addrs = ctx.tensor({batch}, DataType::kDouble);
    Tensor val_buf_addrs = ctx.tensor({batch}, DataType::kDouble);
    key_buf_addrs.from_buffer(key_buf_addresses.data());
    val_buf_addrs.from_buffer(val_buf_addresses.data());

    int max_len_buf = *std::max_element(lens.begin(), lens.end());
    Tensor output = ctx.tensor({batch, len_q, num_kv_heads * m_query, dim_head}, DataType::kHalf);

    cudaEvent_t start, stop;
    auto stream = ctx.current_stream()->ptr;

    int warmup = std::max(1, round / 10);
    for (int i = 0; i < warmup + round; ++i) {
        logger::createStartEvent(i == warmup, &start, &stop, stream);
        if (m_query > 1) {
            nn::multi_query_attention_rag_buffer(
                ctx, q, buf_lens, key_buf_addrs, val_buf_addrs, mask, scale, max_len_buf, output, m_query, algo_id);
        } else if (m_query == 1) {
            nn::attention_qkv_rag_buffer(
                ctx, q, buf_lens, key_buf_addrs, val_buf_addrs, mask, Tensor(), scale, max_len_buf, output);
        } else {
            throw std::runtime_error("invalid m_query");
        }
    }

    float elapsed_ms = logger::destroyDiffEvent(start, stop, stream);
    std::cout << std::setprecision(4)
              << "attention take " << (elapsed_ms * 1000 / float(round)) << "us"
              << ", ALGO=" << algo_id
              << ", batch=" << batch
              << ", len_q=" << len_q
              << ", num_kv_heads=" << num_kv_heads
              << ", len_buf=" << lens
              << std::endl;
}

int main() {
    bmengine::core::Engine engine({
        {0, 1L << 30L},
    });
    auto ctx = engine.create_context();
    auto with_dev = ctx.with_device(0);

    // test_mq_rag_buffer1(ctx, 1, 1, {1024}, 1, 8, 128, 0, 5000);

    for (int b = 1; b <= 5; ++b) {
        for (int len_q = 1; len_q < 2; ++len_q) {
            test_mq_rag_buffer1(ctx, b, len_q, {4096, 123}, 4, 4, 128, 1, 5000);
        }
    }
    // test_self_attention(ctx, 1500, 1536, 4, 0, 2);

    return 0;
}
