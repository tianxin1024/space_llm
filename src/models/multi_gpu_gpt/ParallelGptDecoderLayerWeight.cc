#include "models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"

namespace space_llm {

template <typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int int8_mode) :
    int8_mode_(int8_mode) {
}

template <typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const int hidden_units,
                                                                const int inter_size,
                                                                const int tensor_para_size,
                                                                const int tensor_para_rank,
                                                                const int int8_mode,
                                                                gptVariantParams gpt_variant_params) :
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    int8_mode_(int8_mode),
    gpt_variant_params_(gpt_variant_params) {
    mallocWeights();
    setWeightPtr();

    QK_CHECK_WITH_INFO(!(std::is_same<T, float>::value && int8_mode_ == 1),
                       "Weight only quant does not work with FP32 compute.");
}

template <typename T>
ParallelGptDecoderLayerWeight<T>::~ParallelGptDecoderLayerWeight() {
    if (is_maintain_buffer == true) {
        for (int i = 0; i < weights_ptr.size(); ++i) {
            if (weights_ptr[i] != nullptr) {
                deviceFree(weights_ptr[i]);
            }
        }

        pre_layernorm_weights.beta = nullptr;
        pre_layernorm_weights.gamma = nullptr;
        self_attention_weights.query_weight.kernel = nullptr;
        self_attention_weights.query_weight.bias = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias = nullptr;
        self_attn_layernorm_weights.beta = nullptr;
        self_attn_layernorm_weights.gamma = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias = nullptr;
        ffn_weights.output_weight.kernel = nullptr;
        ffn_weights.output_weight.bias = nullptr;

        after_attention_adapter_weights.intermediate_weight.kernel = nullptr;
        after_attention_adapter_weights.intermediate_weight.bias = nullptr;
        after_attention_adapter_weights.output_weight.kernel = nullptr;
        after_attention_adapter_weights.output_weight.bias = nullptr;

        after_ffn_adapter_weights.intermediate_weight.kernel = nullptr;
        after_ffn_adapter_weights.intermediate_weight.bias = nullptr;
        after_ffn_adapter_weights.output_weight.kernel = nullptr;
        after_ffn_adapter_weights.output_weight.bias = nullptr;

        if (int8_mode_ != 0) {
            // TODO ...
        }

        is_maintain_buffer = false;
    }
}

template <typename T>
void ParallelGptDecoderLayerWeight<T>::copyFrom(const ParallelGptDecoderLayerWeight &other) {
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], hidden_units_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    if (gpt_variant_params_.has_adapters) {
        // Copy adapter biases regardless of int8 mode
        cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[15], other.weights_ptr[15], hidden_units_);
        cudaD2Dcpy(weights_ptr[17], other.weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[19], other.weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ == 0) {
        cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);

        if (gpt_variant_params_.has_adapters) {
            cudaD2Dcpy(weights_ptr[12],
                       other.weights_ptr[12],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(weights_ptr[14],
                       other.weights_ptr[14],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(weights_ptr[16],
                       other.weights_ptr[16],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(weights_ptr[18],
                       other.weights_ptr[18],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }
    } else {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], inter_size_ / tensor_para_size_ * hidden_units_);

        if (gpt_variant_params_.has_adapters) {
            // Copy weights for FFN adapters after attn and regular FFN
            cudaD2Dcpy(int8_weights_ptr[4],
                       other.int8_weights_ptr[4],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[5],
                       other.int8_weights_ptr[5],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            cudaD2Dcpy(int8_weights_ptr[6],
                       other.int8_weights_ptr[6],
                       hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            cudaD2Dcpy(int8_weights_ptr[7],
                       other.int8_weights_ptr[7],
                       gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }

        if (int8_mode_ == 1) {
            cudaD2Dcpy(weight_only_scale_ptr[0], other.weight_only_scale_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[1], other.weight_only_scale_ptr[1], hidden_units_);
            cudaD2Dcpy(weight_only_scale_ptr[2], other.weight_only_scale_ptr[2], inter_size_ / tensor_para_size_);
            cudaD2Dcpy(weight_only_scale_ptr[3], other.weight_only_scale_ptr[3], hidden_units_);

            if (gpt_variant_params_.has_adapters) {
                cudaD2Dcpy(weight_only_scale_ptr[4],
                           other.weight_only_scale_ptr[4],
                           gpt_variant_params_.adapter_inter_size / tensor_para_size_);
                cudaD2Dcpy(weight_only_scale_ptr[5], other.weight_only_scale_ptr[5], hidden_units_);
                cudaD2Dcpy(weight_only_scale_ptr[6],
                           other.weight_only_scale_ptr[6],
                           gpt_variant_params_.adapter_inter_size / tensor_para_size_);
                cudaD2Dcpy(weight_only_scale_ptr[7], other.weight_only_scale_ptr[7], hidden_units_);
            }
        } else if (int8_mode_ == 2) {
            cudaD2Dcpy(scale_ptr[0], other.scale_out_ptr[0], 1);
            cudaD2Dcpy(scale_inter_ptr[0], other.scale_inter_ptr[0], 3 * hidden_units_ / tensor_para_size_);
            cudaD2Dcpy(scale_out_ptr[0], other.scale_out_ptr[0], 3);

            for (int i = 1; i < 4; i++) {
                cudaD2Dcpy(scale_ptr[i], other.scale_ptr[i], 1);
                cudaD2Dcpy(scale_inter_ptr[i], other.scale_inter_ptr[i], 1);
                cudaD2Dcpy(scale_out_ptr[i], other.scale_out_ptr[i], 1);
            }
        }
    }
}

template <typename T>
ParallelGptDecoderLayerWeight<T>::ParallelGptDecoderLayerWeight(const ParallelGptDecoderLayerWeight &other) :
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    int8_mode_(other.int8_mode_),
    gpt_variant_params_(other.gpt_variant_params_) {
    mallocWeights();
    copyFrom(other);
    setWeightPtr();
}

template <typename T>
ParallelGptDecoderLayerWeight<T> &ParallelGptDecoderLayerWeight<T>::operator=(const ParallelGptDecoderLayerWeight &other) {
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    int8_mode_ = other.int8_mode_;
    gpt_variant_params_ = other.gpt_variant_params_;

    mallocWeights();
    copyFrom(other);
    setWeightPtr();

    return *this;
}

template <typename T>
void ParallelGptDecoderLayerWeight<T>::loadModel(std::string dir_path, QKCudaDataType model_file_type) {
    QK_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[1], {hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[3], {3, hidden_units_ / tensor_para_size_},
                         dir_path + ".attention.query_key_value.bias." + std::to_string(tensor_para_rank_) + ".bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[5], {hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[6], {hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[7], {hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[9], {inter_size_},
                         dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[11], {hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin", model_file_type);

    if (gpt_variant_params_.has_adapters) {
        loadWeightFromBin<T>(weights_ptr[13], {gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_attention_adapter.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[15], {hidden_units_},
                             dir_path + ".after_attention_adapter.dense_4h_to_h.bias.bin", model_file_type);
        loadWeightFromBin<T>(weights_ptr[17], {gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                             dir_path + ".after_ffn_adapter.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin",
                             model_file_type);
        loadWeightFromBin<T>(weights_ptr[19], {hidden_units_}, dir_path + ".after_ffn_adapter.dense_4h_to_h.bias.bin", model_file_type);
    }

    // Load weights for GPT
    if (int8_mode_ == 0) {
        loadWeightFromBin<T>(weights_ptr[2], {hidden_units_, 3 * hidden_units_ / tensor_para_size_},
                             dir_path + ".attention.query_key_value.weight." + std::to_string(tensor_para_rank_) + ".bin", model_file_type);

        loadWeightFromBin<T>(weights_ptr[4], {hidden_units_ / tensor_para_size_, hidden_units_},
                             dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_) + ".bin", model_file_type);

        loadWeightFromBin<T>(weights_ptr[8], {hidden_units_, inter_size_ / tensor_para_size_},
                             dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin", model_file_type);

        loadWeightFromBin<T>(weights_ptr[10], {inter_size_ / tensor_para_size_, hidden_units_},
                             dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin", model_file_type);

        // Load adapter weights if required.
        if (gpt_variant_params_.has_adapters) {
            loadWeightFromBin<T>(weights_ptr[12],
                                 {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                                 dir_path + ".after_attention_adapter.dense_h_to_4h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[14],
                                 {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                                 dir_path + ".after_attention_adapter.dense_4h_to_h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[16],
                                 {hidden_units_, gpt_variant_params_.adapter_inter_size / tensor_para_size_},
                                 dir_path + ".after_ffn_adapter.dense_h_to_4h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);

            loadWeightFromBin<T>(weights_ptr[18],
                                 {gpt_variant_params_.adapter_inter_size / tensor_para_size_, hidden_units_},
                                 dir_path + ".after_ffn_adapter.dense_4h_to_h.weight."
                                     + std::to_string(tensor_para_rank_) + ".bin",
                                 model_file_type);
        }
    } else if (int8_mode_ == 1) {
        // TODO ...
    } else if (int8_mode_ == 2) {
        // TODO ...
    }
}

template <typename T>
void ParallelGptDecoderLayerWeight<T>::mallocWeights() {
    deviceMalloc(&weights_ptr[0], hidden_units_);                         // pre layer norm beta
    deviceMalloc(&weights_ptr[1], hidden_units_);                         // pre layer norm gamma
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_); // qkv biases
    deviceMalloc(&weights_ptr[5], hidden_units_);                         // attention output bias
    deviceMalloc(&weights_ptr[6], hidden_units_);                         // attn layer norm beta
    deviceMalloc(&weights_ptr[7], hidden_units_);                         // attn layer norm gamma
    deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);       // ffn inter bias
    deviceMalloc(&weights_ptr[11], hidden_units_);                        // ffn output bais

    // Alloc biases adapters. They do not get quantized so are placed here.
    if (gpt_variant_params_.has_adapters) {
        deviceMalloc(&weights_ptr[13], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[15], hidden_units_);
        deviceMalloc(&weights_ptr[17], gpt_variant_params_.adapter_inter_size / tensor_para_size_);
        deviceMalloc(&weights_ptr[19], hidden_units_);
    }

    if (int8_mode_ == 0) {
        deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_); // qkv weights
        deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);     // attention output weight
        deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);       // ffn inter weight
        deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);      // ffn output weight

        // Alloc weights for adapters
        if (gpt_variant_params_.has_adapters) {
            deviceMalloc(&weights_ptr[12], hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&weights_ptr[14], gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
            deviceMalloc(&weights_ptr[16], hidden_units_ * gpt_variant_params_.adapter_inter_size / tensor_para_size_);
            deviceMalloc(&weights_ptr[18], gpt_variant_params_.adapter_inter_size / tensor_para_size_ * hidden_units_);
        }
    } else {
        // Alloc FFN and Attention int8 weights
        // TODO ...
    }
}

template <typename T>
void ParallelGptDecoderLayerWeight<T>::setWeightPtr() {
    pre_layernorm_weights.beta = weights_ptr[0];
    pre_layernorm_weights.gamma = weights_ptr[1];
    self_attention_weights.query_weight.kernel = weights_ptr[2];
    self_attention_weights.query_weight.bias = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias = weights_ptr[5];
    self_attn_layernorm_weights.beta = weights_ptr[6];
    self_attn_layernorm_weights.gamma = weights_ptr[7];

    ffn_weights.intermediate_weight.kernel = weights_ptr[8];
    ffn_weights.intermediate_weight.bias = weights_ptr[9];
    ffn_weights.output_weight.kernel = weights_ptr[10];
    ffn_weights.output_weight.bias = weights_ptr[11];

    after_attention_adapter_weights.intermediate_weight.kernel = weights_ptr[12];
    after_attention_adapter_weights.intermediate_weight.bias = weights_ptr[13];
    after_attention_adapter_weights.output_weight.kernel = weights_ptr[14];
    after_attention_adapter_weights.output_weight.bias = weights_ptr[15];

    after_ffn_adapter_weights.intermediate_weight.kernel = weights_ptr[16];
    after_ffn_adapter_weights.intermediate_weight.bias = weights_ptr[17];
    after_ffn_adapter_weights.output_weight.kernel = weights_ptr[18];
    after_ffn_adapter_weights.output_weight.bias = weights_ptr[19];

    if (int8_mode_ != 0) {
        // TODO ...
    }

    is_maintain_buffer = true;
}

template struct ParallelGptDecoderLayerWeight<float>;
template struct ParallelGptDecoderLayerWeight<half>;

} // namespace space_llm
