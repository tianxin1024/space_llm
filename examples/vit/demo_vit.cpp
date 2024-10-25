#include <iostream>
#include "models/vit/ViT.h"
#include <cuda_profiler_api.h>

using namespace space_llm;

template <typename T>
void vit_inference(int batch_size, int img_size, int patch_size, int embed_dim, int head_num, int layer_num, int token_classifier) {
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream = 0;
    checkCUDNN(cudnnCreate(&cudnn_handle));
    checkCUDNN(cudnnSetStream(cudnn_handle, stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));

    cublasAlgoMap *cublas_algo_map = new cublasAlgoMap("gemm_config.in");
    std::mutex *cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper *cublas_wrapper = new cublasMMWrapper(cublas_handle, cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, nullptr);

    int max_batch = batch_size;
    const int in_chans = 3;
    const int inter_size = embed_dim * 4;
    const bool with_cls_token = token_classifier > 0;

    AttentionType attention_type = AttentionType::UNFUSED_MHA;
    Allocator allocator(0);

    ViTTransformer<T> *vit = new ViTTransformer<T>(max_batch,
                                                   img_size,
                                                   in_chans,
                                                   patch_size,
                                                   embed_dim,
                                                   head_num,
                                                   inter_size,
                                                   layer_num,
                                                   with_cls_token,
                                                   86, // sm
                                                   1.0f,
                                                   stream,
                                                   cudnn_handle,
                                                   cublas_wrapper,
                                                   &allocator,
                                                   false,
                                                   attention_type);
}

int main() {
    int batch_size = 1;
    int img_size = 224;
    int patch_size = 16;
    int embed_dim = 768;
    int head_num = 12;
    int layer_num = 12;
    int token_classifier = 1;

    vit_inference<float>(batch_size, img_size, patch_size, embed_dim, head_num, layer_num, token_classifier);
    return 0;
}
