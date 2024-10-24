#pragma once

#include "utils/denseWeight.h"
#include "models/vit/ViTLayerWeight.h"

namespace space_llm {

#define WEIGHT_N 6

template <typename T>
struct ViTEmbeds {
    const T *class_embed;
    const T *position_embed;
};

template <typename T>
struct ViTWeight {
    ViTWeight() = delete;
    ViTWeight(const int embed_dim,
              const int inter_size,
              const int num_layer,
              const int img_size,
              const int patch_size,
              const int chn_num,
              const bool with_cls_token,
              const bool hold_buffer = true) :
        with_cls_token_(with_cls_token),
        embed_dim_(embed_dim),
        inter_size_(inter_size),
        num_layer_(num_layer),
        img_size_(img_size),
        patch_size_(patch_size),
        chn_num_(chn_num),
        seq_len_(img_size_ * img_size_ / (patch_size_ * patch_size_) + (with_cls_token_ ? 1 : 0)) {
        weights_size[0] = chn_num_ * patch_size_ * patch_size_ * embed_dim_;
        weights_size[1] = embed_dim_;
        weights_size[2] = with_cls_token_ ? embed_dim_ : 0;
        weights_size[3] = embed_dim_ * seq_len_;
        weights_size[4] = embed_dim_;
        weights_size[5] = embed_dim_;

        if (hold_buffer) {
            for (int i = 0; i < WEIGHT_N; ++i) {
                if (weights_size[i] == 0) {
                    continue;
                }

                deviceMalloc(&weights_ptr[i], weights_size[i]);
            }

            setWeightPtr();
        }

        vit_layer_weights.reserve(num_layer_);
        for (int i = 0; i < num_layer_; ++i) {
            vit_layer_weights.push_back(ViTLayerWeight<T>(embed_dim_, inter_size_, i, hold_buffer));
        }
    }

    ~ViTWeight() {
        if (is_maintain_buffer == true) {
            vit_layer_weights.clear();
            for (int i = 0; i < WEIGHT_N; ++i) {
                if (weights_ptr[i] != nullptr) {
                    deviceFree(weights_ptr[i]);
                }
            }

            post_transformer_layernorm_weights.gamma = nullptr;
            post_transformer_layernorm_weights.beat = nullptr;
            pre_transform_embeds.class_embed = nullptr;
            pre_transform_embeds.position_embed = nullptr;
            pre_encoder_conv_weights.kernel = nullptr;
            pre_encoder_conv_weights.bias = nullptr;
            is_maintain_buffer = false;
        }
    }

    std::vector<ViTLayerWeight<T>> vit_layer_weights;
    DenseWeight<T> pre_encoder_conv_weights;
    ViTEmbeds<T> pre_transform_embeds;
    LayerNormWeight<T> post_transformer_layernorm_weights;
    bool with_cls_token_;

private:
    void setWeightPtr() {
        pre_encoder_conv_weights.kernel = weights_ptr[0];
        pre_encoder_conv_weights.bias = weights_ptr[1];
        pre_transform_embeds.class_embed = weights_ptr[2];
        pre_transform_embeds.position_embed = weights_ptr[3];
        post_transformer_layernorm_weights.gamma = weights_ptr[4];
        post_transformer_layernorm_weights.beta = weights_ptr[5];

        is_maintain_buffer = true;
    }
    int embed_dim_;
    int inter_size_;
    int num_layer_;
    int img_size_;
    int patch_size_;
    int chn_num_;
    int seq_len_;
    bool is_maintain_buffer = false;
    T *weights_ptr[WEIGHT_N]{nullptr};
    size_t weights_size[WEIGHT_N];
}; // struct ViTWeight

#undef WEIGHT_N

} // namespace space_llm
