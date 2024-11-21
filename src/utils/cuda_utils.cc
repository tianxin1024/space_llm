#include "utils/cuda_utils.h"
#include "utils/INIReader.h"

namespace space_llm {

/* **************************** debug tools ********************************* */

template <typename T>
void print_abs_mean(const T *buf, uint size, cudaStream_t stream, std::string name) {
    if (buf == nullptr) {
        QK_LOG_ERROR("It is an nullptr, skip!");
        return;
    }

    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    T *h_tmp = new T[size];
    cudaMemcpyAsync(h_tmp, buf, sizeof(T) * size, cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    double sum = 0.0f;
    uint64_t zero_count = 0;
    float max_val = -1e10;
    bool find_inf = false;
    for (uint i = 0; i < size; i++) {
        if (std::isinf((float)(h_tmp[i]))) {
            find_inf = true;
            continue;
        }
        sum += abs((double)h_tmp[i]);
        if ((float)h_tmp[i] == 0.0f) {
            zero_count++;
        }
        max_val = max_val > abs(float(h_tmp[i])) ? max_val : abs(float(h_tmp[i]));
    }
    printf("[INFO][QK] %20s size: %u, abs mean: %f, abs sum: %f, abs max: %f, find inf: %s",
           name.c_str(),
           size,
           sum / size,
           sum,
           max_val,
           find_inf ? "true" : "false");
    std::cout << std::endl;
    delete[] h_tmp;
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
}

template void print_abs_mean(const float *buf, uint size, cudaStream_t stream, std::string name);
template void print_abs_mean(const half *buf, uint size, cudaStream_t stream, std::string name);
template void print_abs_mean(const int *buf, uint size, cudaStream_t stream, std::string name);
template void print_abs_mean(const uint *buf, uint size, cudaStream_t stream, std::string name);
template void print_abs_mean(const int8_t *buf, uint size, cudaStream_t stream, std::string name);

template <typename T>
void print_to_screen(const T *result, const int size) {
    if (result == nullptr) {
        QK_LOG_WARNING("It is an nullptr, skip! \n");
        return;
    }
    T *tmp = reinterpret_cast<T *>(malloc(sizeof(T) * size));
    check_cuda_error(cudaMemcpy(tmp, result, sizeof(T) * size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        printf("%d, %f\n", i, static_cast<float>(tmp[i]));
    }
    free(tmp);
}

template void print_to_screen(const float *result, const int size);
template void print_to_screen(const half *result, const int size);
template void print_to_screen(const int *result, const int size);
template void print_to_screen(const uint *result, const int size);
template void print_to_screen(const bool *result, const int size);

/* ***************************** common utils ****************************** */

cudaError_t getSetDevice(int i_device, int *o_device) {
    int current_dev_id = 0;
    cudaError_t err = cudaSuccess;

    if (o_device != NULL) {
        err = cudaGetDevice(&current_dev_id);
        if (err != cudaSuccess) {
            return err;
        }
        if (current_dev_id == i_device) {
            *o_device = i_device;
        } else {
            err = cudaSetDevice(i_device);
            if (err != cudaSuccess) {
                return err;
            }
            *o_device = current_dev_id;
        }
    } else {
        err = cudaSetDevice(i_device);
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}

QKCudaDataType getModelFileType(std::string ini_file, std::string section_name) {
    QKCudaDataType model_file_type;
    INIReader reader = INIReader(ini_file);
    if (reader.ParseError() < 0) {
        QK_LOG_WARNING("Can't load %s. Use FP32 as default", ini_file.c_str());
        model_file_type = QKCudaDataType::FP32;
    } else {
        std::string weight_data_type_str = std::string(reader.Get(section_name, "weight_data_type"));
        if (weight_data_type_str.find("fp32") != std::string::npos) {
            model_file_type = QKCudaDataType::FP32;
        } else if (weight_data_type_str.find("fp16") != std::string::npos) {
            model_file_type = QKCudaDataType::FP16;
        } else if (weight_data_type_str.find("bf16") != std::string::npos) {
            model_file_type = QKCudaDataType::BF16;
        } else {
            QK_LOG_WARNING("Invalid type %s. Use FP32 as default", weight_data_type_str.c_str());
            model_file_type = QKCudaDataType::FP32;
        }
    }
    return model_file_type;
}
/* ************************** end of common utils ************************** */

} // namespace space_llm
