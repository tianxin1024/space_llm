#include "layers/sampling_layers/BaseSamplingLayer.h"

namespace space_llm {

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(size_t max_batch_size,
                                        size_t vocab_size,
                                        size_t vocab_size_padded,
                                        int end_id,
                                        size_t top_k,
                                        float top_p,
                                        unsigned long long random_seed,
                                        float temperature,
                                        float len_penalty,
                                        float repetition_penalty,
                                        cudaStream_t stream,
                                        cublasMMWrapper *cublas_wrapper,
                                        IAllocator *allocator,
                                        bool is_free_buffer_after_forward,
                                        cudaDeviceProp *cuda_device_prop) :
    DynamicDecodeBaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    vocab_size_(vocab_size),
    vocab_size_padded_(vocab_size_padded) {
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(BaseSamplingLayer const &sampling_layer) :
    DynamicDecodeBaseLayer(sampling_layer),
    vocab_size_(sampling_layer.vocab_size_),
    vocab_size_padded_(sampling_layer.vocab_size_padded_),
    sampling_workspace_size_(sampling_layer.sampling_workspace_size_) {
}

template <typename T>
BaseSamplingLayer<T>::~BaseSamplingLayer() {
}

template <typename T>
void BaseSamplingLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap *runtime_args) {
    // Set up the sampling layer for given runtime arguments.
    //
    // runtime_args:
    //      runtime_top_k [1] or [batch_size] on cpu, optional.
    //      runtime_top_p [1] or [batch_size] on cpu, optional.
    //      temperature [1] or [batch_size] on cpu, optional.
    //      repetition_penalty [1] or [batch_size] on cpu, optional.
    //      presence_penalty [1] or [batch_size] on cpu, optional.
    //          repetition_penalty and presence_penalty are multually exclusive.
    //      min_length [1] or [batch_size] on cpu, optional.

    QK_LOG_DEBUG(__PRETTY_FUNCTION__);
    Tensor runtime_top_k = runtime_args->isExist("runtime_top_k") ? runtime_args->at("runtime_top_k") : Tensor();
    Tensor runtime_top_p = runtime_args->isExist("runtime_top_p") ? runtime_args->at("runtime_top_p") : Tensor();
    allocateBuffer(batch_size, runtime_top_k, runtime_top_p);

    // If runtime argument has single random seed, using this random seed to initialize the random table of all
    // sentences. If the argument has [batch_size] random seeds, initializing the random table by different random seeds
    // respectively. If no random seed, initialize the random table of all sentences by 0 directly.
    if (runtime_args->isExist("random_seed")) {
        Tensor random_seeds = runtime_args->at("random_seed");
        QK_CHECK_WITH_INFO(random_seeds.shape.size() == 1
                               && (random_seeds.size() == 1 || random_seeds.size() == batch_size),
                           fmtstr("random_seeds must be of shape [1] or [batch_size(%ld)], got random_seeds.shape=%s",
                                  batch_size, vec2str(random_seeds.shape).c_str()));
        if (random_seeds.size() == 1) {
            invokeCurandInitialize(curandstate_buf_, batch_size, random_seeds.getVal<unsigned long long>(), stream_);
            sync_check_cuda_error();
        } else {
            unsigned long long *random_seed_ptr = random_seeds.getPtr<unsigned long long>();
            cudaAutoCpy(random_seeds_buf_, random_seed_ptr, batch_size, stream_);
            invoke_CurandBatchInitialize(curandstate_buf_, batch_size, random_seeds_buf_, stream_);
            sync_check_cuda_error();
        }
    } else {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize(curandstate_buf_, batch_size, 0, stream_);
    }

    // Setup penalties.
    const float default_temperature = 1.0f;
    Tensor temperature = runtime_args->isExist("temperature") ?
                             runtime_args->at("temperature") :
                             Tensor(MEMORY_CPU, TYPE_FP32, {1}, &default_temperature);
    if (temperature.size() == 1) {
        float tp = temperature.getVal<float>();
        deviceFill(temperature_buf_, batch_size, tp, stream_);
        std::fill_n(temperature_, batch_size, tp);
    } else {
        cudaAutoCpy(temperature_buf_, temperature.getPtr<float>(), batch_size, stream_);
        std::copy_n(temperature.getPtr<float>(), batch_size, temperature_);
    }

    if (runtime_args->isExist("presence_penalty") && runtime_args->isExist("repetition_penalty")) {
        QK_CHECK_WITH_INFO(
            !(runtime_args->isExist("repetition_penalty") && runtime_args->isExist("presence_penalty")),
            "Found ambiguous parameters repetition_penalty and presence_penalty which are mutually exclusive. "
            "Please provide one of repetition_penalty or presence_penalty.");
        // TODO ...
    }
    // TOTO ...
}
template <typename T>
void BaseSamplingLayer<T>::forward(std::vector<Tensor> *output_tensors, const std::vector<Tensor> *input_tensors) {
}
template <typename T>
void BaseSamplingLayer<T>::forward(std::unordered_map<std::string, Tensor> *output_tensors,
                                   const std::unordered_map<std::string, Tensor> *input_tensors) {
}
template <typename T>
void BaseSamplingLayer<T>::forward(TensorMap *output_tensors, TensorMap *input_tensors) {
}
} // namespace space_llm