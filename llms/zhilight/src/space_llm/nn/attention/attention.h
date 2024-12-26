#pragma once
#include <core/core.h>
#include "model/model_config.hpp"
#include "kvcache/kvcache.h"

namespace model {
class ModelContext;
}

namespace nn {
using namespace kvcache;
using namespace bmengine;

class Linear;

class Attention : public core::Layer {
    BM_LAYER_DEF(Attention);

    Attention(
        const core::Context &ctx,
        model::ModelConfig block_config,
        model::QuantConfig quant_config,
        bool parallel);

    core::Tensor forward(
        const core::Context &ctx,
        const core::Tensor &inp,  // (len_q, dim_model)
        const core::Tensor &mask, // (len_q, len_buf)
        const core::Tensor &
            position_bias,               // if relative (num_head, len_q, len_buf) else if rotary (len_q)
        const core::Tensor &seqlens_q,   // (batch?, 1,)    int32
        const core::Tensor &seqlens_kv,  // (batch?, 1,)    int32
        const core::Tensor *past_k,      // (num_head, len_buf, dim_head)
        const core::Tensor *past_v,      // (num_head, len_buf, dim_head)
        const core::Tensor *block_table, // (batch_size, blocks_per_seq)
        const core::Tensor *placement,   // (batch?. len_q,)    int32
        core::Tensor *output);

    core::Tensor dyn_rag_forward(
        model::ModelContext &ctx,
        const core::Tensor &inp,      // (grouped_len_q, dim_model)
        const core::Tensor &position, // (grouped_len_q)
        core::Tensor *output = nullptr);

    const Linear &att_out() const;

    void load_state_dict(
        const core::Context &ctx,
        const std::map<std::string, const core::Tensor> &state_dict,
        const std::string &prefix,
        bool allow_missing = false) override;
};

} // namespace nn
