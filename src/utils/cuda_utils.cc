#include "utils/cuda_utils.h"

namespace space_llm {

/* **************************** debug tools ********************************* */

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

} // namespace space_llm
