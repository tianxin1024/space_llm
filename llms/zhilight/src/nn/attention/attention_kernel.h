#pragma once

#include "core/core.h"

namespace nn {
using namespace bmengine;

struct AttentionWorkspace {
    core::Tensor cache;
    core::Tensor local_max;
    core::Tensor local_sum_exp;
};

AttentionWorkspace get_mqa_workspace(
    const core::Context &ctx,
    const core::Tensor &batch_q,
    int max_len_buf,
    bool is_quantized);

void multi_query_attention_rag_buffer(
    const core::Context &ctx,
    const core::Tensor &batch_q,       // (batch, len_q, num_kv_heads * m_query, dim_head)
    const core::Tensor &buf_lens,      // (batch)
    const core::Tensor &key_buf_addrs, // (batch) => (num_kv_heads, len_buf, dim_head)
    const core::Tensor &val_buf_addrs, // (batch) => (num_kv_heads, len_buf, dim_head)
    const core::Tensor &mask,          // (batch) => (len_q, len_buf)
    const float scale,
    const int max_len_buf,
    core::Tensor &output, // (batch, len_q, num_kv_heads * m_query, dim_head)
    const int m_query = 8,
    int algo_id = -1,
    const AttentionWorkspace &ws = {},
    const core::Tensor &scale_key_addrs = core::Tensor(),
    const core::Tensor &scale_val_addrs = core::Tensor(),
    core::DataType dequant_dtype = core::DataType::kHalf);

} // namespace nn
