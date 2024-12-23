#pragma once
#include "core/core.h"

namespace nn {
using namespace bmengine;

void softmax(
    const core::Context &ctx,
    const core::Tensor &logits,
    const core::Tensor &output,
    float temperature);

std::tuple<float, core::Tensor, core::Tensor> cross_entropy(
    const core::Context &ctx,
    const std::tuple<core::Tensor, core::Tensor> &
        logits_tuple,           // ((seq_len, len_vocab), (seq_len, len_ext))
    const core::Tensor &labels, // (seq_len)
    int32_t ignore_index,       // -100
    float loss_scale            // 1.0
);

int greedy_match(
    const core::Context &ctx,
    const std::tuple<core::Tensor, core::Tensor> &
        logits_tuple,           // ((seq_len, len_vocab), (seq_len, len_ext))
    const core::Tensor &labels, // (seq_len)
    int32_t ignore_index        // -100
);

std::tuple<core::Tensor, core::Tensor> log_prob(
    const core::Context &ctx,
    const std::tuple<core::Tensor, core::Tensor> &
        logits_tuple,           // ((seq_len, len_vocab), (seq_len, len_ext))
    const core::Tensor &labels, // (seq_len)
    int32_t ignore_index        // -100
);

std::tuple<float, core::Tensor> cross_entropy_raw(
    const core::Context &ctx,
    const core::Tensor &logits, // (seq_len, len_vocab)
    const core::Tensor &labels, // (seq_len)
    int32_t ignore_index,       // -100
    float loss_scale            // 1.0
);

int greedy_match_raw(
    const core::Context &ctx,
    const core::Tensor &logits, // (seq_len, len_vocab)
    const core::Tensor &labels, // (seq_len)
    int32_t ignore_index        // -100
);

std::tuple<float, core::Tensor> log_prob_raw(
    const core::Context &ctx,
    const core::Tensor &logits, // (seq_len, len_vocab)
    const core::Tensor &labels, // (seq_len)
    int32_t ignore_index        // -100
);

void multiply(const core::Context &ctx, const core::Tensor &a, float b, core::Tensor *c);

void attn_softmax(
    const core::Context &ctx,
    float scale,
    const core::Tensor &attn_score, // (batch, num_heads, len_q, len_buf)
    const core::Tensor &mask,       // (batch, len_q, len_buf)
    const core::Tensor &
        position_bias // if relative (batch, num_head, len_q, len_buf) else if core::Tensor()
);

} // namespace nn
