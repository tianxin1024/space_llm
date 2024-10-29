#pragma once

#include "layers/attention_layers/AttentionWeight.h"
#include "kernels/layernorm_kernels.h"
#include "layers/ffnWeight.h"
#include "utils/memory_utils.h"

namespace space_llm {

#define WEIGHT_N 16

template <typename T>
struct ViTLayerWeight {
    ViTLayerWeight() = delete;
    ViTLayerWeight(const int embed_dim, const int inter_size, int layer_idx, const bool hold_buffer) :
        embed_dim_(embed_dim), inter_size_(inter_size), layer_idx_(layer_idx) {
        weights_size[0] = embed_dim_ * embed_dim_;
        weights_size[1] = embed_dim_;
        weights_size[2] = embed_dim_ * embed_dim_;
        weights_size[3] = embed_dim_;
        weights_size[4] = embed_dim_ * embed_dim_;
        weights_size[5] = embed_dim_;
        weights_size[6] = embed_dim_ * embed_dim_;
        weights_size[7] = embed_dim_;
        weights_size[8] = embed_dim_;
        weights_size[9] = embed_dim_;
        weights_size[10] = embed_dim_ * inter_size_;
        weights_size[11] = inter_size_;
        weights_size[12] = inter_size_ * embed_dim_;
        weights_size[13] = embed_dim_;
        weights_size[14] = embed_dim_;
        weights_size[15] = embed_dim_;

        if (hold_buffer) {
            for (int i = 0; i < WEIGHT_N; ++i) {
                deviceMalloc(&weights_ptr[i], weights_size[i]);
            }

            setWeightPtr();
        }
    }

    ~ViTLayerWeight() {
        if (is_maintain_buffer == true) {
            for (int i = 0; i < WEIGHT_N; ++i) {
                deviceFree(weights_ptr[i]);
            }
            attention_weights.query_weight.kernel = nullptr;
            attention_weights.query_weight.bias = nullptr;
            attention_weights.key_weight.kernel = nullptr;
            attention_weights.key_weight.bias = nullptr;
            attention_weights.value_weight.kernel = nullptr;
            attention_weights.value_weight.bias = nullptr;
            attention_weights.attention_output_weight.kernel = nullptr;
            attention_weights.attention_output_weight.bias = nullptr;
            attn_layernorm_weights.gamma = nullptr;
            attn_layernorm_weights.beta = nullptr;
            ffn_weights.intermediate_weight.kernel = nullptr;
            ffn_weights.intermediate_weight.bias = nullptr;
            ffn_weights.output_weight.kernel = nullptr;
            ffn_weights.output_weight.bias = nullptr;
            ffn_layernorm_weights.gamma = nullptr;
            ffn_layernorm_weights.beta = nullptr;
            is_maintain_buffer = false;
        }
    }

    ViTLayerWeight(const ViTLayerWeight &other) :
        embed_dim_(other.embed_dim_), inter_size_(other.inter_size_) {
        memcpy(weights_size, other.weights_size, sizeof(size_t) * WEIGHT_N);
        layer_idx_ = other.layer_idx_;
        if (other.is_maintain_buffer) {
            for (int i = 0; i < WEIGHT_N; ++i) {
                if (!is_maintain_buffer) {
                    deviceMalloc(&weights_ptr[i], weights_size[i]);
                }
                cudaD2Dcpy(weights_ptr[i], other.weights_ptr[i], weights_size[i]);
            }
            setWeightPtr();
        }
    }

    AttentionWeight<T> attention_weights;
    LayerNormWeight<T> attn_layernorm_weights;
    ffnWeight<T> ffn_weights;
    LayerNormWeight<T> ffn_layernorm_weights;

private:
    void setWeightPtr() {
        attention_weights.query_weight.kernel = weights_ptr[0];
        attention_weights.query_weight.bias = weights_ptr[1];
        attention_weights.key_weight.kernel = weights_ptr[2];
        attention_weights.key_weight.bias = weights_ptr[3];
        attention_weights.value_weight.kernel = weights_ptr[4];
        attention_weights.value_weight.bias = weights_ptr[5];
        attention_weights.attention_output_weight.kernel = weights_ptr[6];
        attention_weights.attention_output_weight.bias = weights_ptr[7];
        attn_layernorm_weights.gamma = weights_ptr[8];
        attn_layernorm_weights.beta = weights_ptr[9];
        ffn_weights.intermediate_weight.kernel = weights_ptr[10];
        ffn_weights.intermediate_weight.bias = weights_ptr[11];
        ffn_weights.output_weight.kernel = weights_ptr[12];
        ffn_weights.output_weight.bias = weights_ptr[13];
        ffn_layernorm_weights.gamma = weights_ptr[14];
        ffn_layernorm_weights.beta = weights_ptr[15];

        is_maintain_buffer = true;
    }

    int embed_dim_;
    int inter_size_;
    int layer_idx_;
    bool is_maintain_buffer = false;
    T *weights_ptr[WEIGHT_N]{nullptr};
    size_t weights_size[WEIGHT_N];
    bool is_maintain_sp_buffer = false;

}; // struct ViTLayerWeight

#undef WEIGHT_N

} // namespace space_llm
