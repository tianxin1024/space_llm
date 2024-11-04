
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// The structure of parameters for the masked multihead attention kernel.
//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.

template <typename T>
struct Multihead_attention_params_base {
    // The output buffer. Dimensions B x D.
    T *out = nullptr;

    // The input Qs and the associated bias. Dimensions B x D and D, resp.
    const T *q = nullptr, *q_bias = nullptr;
    // The input Ks and the associated bias. Dimensions B x D and D, resp.
    const T *k = nullptr, *k_bias = nullptr;
    // The input Vs and the associated bias. Dimensions B x D and D, resp.
    const T *v = nullptr, *v_bias = nullptr;

    // The cache for the Ks. The size must be at least B x L x D.
    T *k_cache = nullptr;
    // The cache for the Vs. The size must be at least B x L x D.
    T *v_cache = nullptr;
    // The indirections to use for cache when beam sampling.
    const int *cache_indir = nullptr;

    // scales
    const float *query_weight_output_scale = nullptr;
    const float *attention_qk_scale = nullptr;
    const float *attention_output_weight_input_scale_inv = nullptr;

    // Stride to handle the case when KQV is a single buffer
    int stride = 0;

    // The batch size.
    int batch_size = 0;
    // The beam width
    int beam_width = 0;
    // The sequence length.
    int memory_max_len = 0;
    // The number of heads (H).
    int num_heads = 0;
    // The hidden dimension per head (Dh).
    int hidden_size_per_head = 0;
    // The per-head latent space reserved for rotary embeddings.
    int rotary_embedding_dim = 0;
    bool neox_rotary_style = false;
    // The maximum length of input sentences.
    int max_input_length = 0;
    // The current timestep. TODO(bhsueh) Check that do we only this param in cross attention?
    int timestep = 0;
    // The current timestep of each sentences (support different timestep for different sentences)

    // The 1.f / sqrt(Dh). Computed on the host.
    float inv_sqrt_dh = 0.0f;

    // Used when we have some input context like gpt
    const int *total_padding_tokens = nullptr;

    const bool *masked_tokens = nullptr;
    const int *prefix_prompt_lengths = nullptr;
    int max_prefix_prompt_length = 0;

    const T *relative_attention_bias = nullptr;
    int relative_attention_bias_stride = 0;
    // The slope per head of linear position bias to attention score (H).
    const T *linear_bias_slopes = nullptr;

    const T *ia3_key_weights = nullptr;
    const T *ia3_value_weights = nullptr;
    const int *ia3_tasks = nullptr;

    const float *qkv_scale_out = nullptr;
    const float *attention_out_scale = nullptr;
    int int8_mode = 0;
};

template <typename T, bool CROSS_ATTENTION>
struct Multihead_attention_params : public Multihead_attention_params_base<T> {
    // output cross attentions
    float *cross_attention_out = nullptr;
    int max_decoder_seq_len = 0;
    bool is_return_cross_attentions = false;

    // allows to exist attention eary
    bool *finished = nullptr;

    // required in case of cross attention
    // will need it here till if constexpr in c++17
    int *memory_length_per_sample = nullptr;

    // required in case of masked attention with different length
    const int *length_per_sample = nullptr;
};

template <typename T>
struct Multihead_attention_params<T, true> : public Multihead_attention_params_base<T> {
    // output cross attentions
    float *cross_attention_out = nullptr;
    int max_decoder_seq_len = 0;
    bool is_return_cross_attentions = false;

    // allows to exist attention eary
    bool *finished = nullptr;

    // required in case of cross attention
    int *memory_length_per_sample = nullptr;

    // required in case of masked attention with different length
    const int *length_per_sample = nullptr;
};

template <class T>
using Masked_multihead_attention_params = Multihead_attention_params<T, false>;
