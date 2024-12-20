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

    if (std::is_same<T, half>::value) {
        cublas_wrapper->setFP16GemmConfig();
    } else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    int max_batch = batch_size;
    const int in_chans = 3;
    const bool with_cls_token = token_classifier > 0;
    const int inter_size = embed_dim * 4;
    const int head_dim = embed_dim / head_num;
    const int seq_len = (img_size / patch_size) * (img_size / patch_size) + (with_cls_token ? 1 : 0);

    ViTWeight<T> params = ViTWeight<T>(embed_dim, inter_size, layer_num, img_size, patch_size, in_chans, with_cls_token);

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

    T *input_d, *output_d;
    deviceMalloc(&input_d, batch_size * img_size * img_size * in_chans, false);
    deviceMalloc(&output_d, batch_size * seq_len * embed_dim, false);

    std::vector<Tensor> input_tensors = std::vector<Tensor>{
        Tensor{MEMORY_GPU,
               getTensorType<T>(),
               std::vector<size_t>{(size_t)batch_size, (size_t)in_chans, (size_t)img_size, (size_t)img_size},
               input_d}};

    std::vector<Tensor> output_tensors =
        std::vector<Tensor>{Tensor{MEMORY_GPU,
                                   getTensorType<T>(),
                                   std::vector<size_t>{(size_t)batch_size, (size_t)seq_len, (size_t)embed_dim},
                                   output_d}};

    // warmup
    for (int i = 0; i < 10; i++) {
        vit->forward(&output_tensors, &input_tensors, &params);
    }

    int ite = 100;
    CudaTimer cuda_timer(stream);
    cuda_timer.start();
    for (int i = 0; i < ite; i++) {
        vit->forward(&output_tensors, &input_tensors, &params);
    }
    float total_time = cuda_timer.stop();

    QK_LOG_INFO("batch_size: %d, img_size : %d,\n"
                "patch_size: %d, embed_dim: %d,\n"
                "head_num  : %d, head_dim : %d,\n"
                "layer_num : %d, is_fp16  : %d,\n"
                "QK-CPP-time %.2f ms (%d iterations) ",
                batch_size,
                img_size,
                patch_size,
                embed_dim,
                head_num,
                head_dim,
                layer_num,
                std::is_same<T, half>::value,
                total_time / ite,
                ite);

    delete vit;
    delete cublas_algo_map;
    delete cublas_wrapper_mutex;

    // free data
    check_cuda_error(cudaFree(output_d));
    check_cuda_error(cudaFree(input_d));
    check_cuda_error(cublasDestroy(cublas_handle));
    check_cuda_error(cublasLtDestroy(cublaslt_handle));
    checkCUDNN(cudnnDestroy(cudnn_handle));

    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
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
