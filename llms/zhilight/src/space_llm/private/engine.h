#pragma once
#include "core/engine.h"
#include "core/thread_pool.h"
#include "core/dtype.h"
#include <mutex>
#include <stack>
#include <nccl.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include "private/allocator.h"
#include "private/stream.h"

namespace bmengine {

namespace core {

class DeviceHandles {
public:
    int dev_id;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    ncclComm_t comm;
    int rank;
    int compute_capability;
    int mp_count;
    int l2_cache_size;
    int max_shared_memory;

    DeviceHandles(int dev_id, ncclUniqueId uniqueID, int rank, int world_size);
    ~DeviceHandles();
    DeviceHandles(const DeviceHandles &) = delete;
    DeviceHandles &operator=(const DeviceHandles &) = delete;
    DeviceHandles(DeviceHandles &&) = default;
    DeviceHandles &operator=(DeviceHandles &&) = delete;
};

class Tensor;
class Context;
class EngineImpl {
    friend class Engine;
    std::vector<DeviceHandles *> handles;
    std::vector<MemoryAllocator *> allocators;
    std::vector<StreamAllocator *> streams;
    std::vector<std::mutex *> device_lock;
    std::vector<TaskThreadPool *> device_threads;
    // for nccl
    std::vector<ncclUniqueId> uniqueIDs;
    int world_size_;

    int debug;
    bool is_mem_frozen{false};

public:
    EngineImpl(const std::vector<DeviceConfiguration> &cfg, int tp);
    ~EngineImpl();
    EngineImpl(const EngineImpl &) = delete;
    EngineImpl &operator=(const EngineImpl &) = delete;
    EngineImpl(EngineImpl &&) = delete;
    EngineImpl &operator=(EngineImpl &&) = delete;

    Context create_context(const std::vector<int> &devices) const;

    /* Thread-safe API */
    DeviceHandles *get_device_handle(int dev_id);
    void alloc_device(int dev_id);
    void release_device(int dev_id);
    cudaStream_t create_stream(int dev_id);
    void destroy_stream(int dev_id, cudaStream_t stream);

    MemoryAllocator *get_allocator(int dev_id) {
        return allocators[dev_id];
    }
    Memory alloc_memory(int dev_id, size_t size, size_t round_up_bytes = 512);
    Tensor alloc_tensor(int dev_id, const std::vector<size_t> &shape, DataType dtype);
    void get_parameter(const std::string &name, Tensor *tensor);
    void init_parameter(const std::string &name, Tensor *tensor);

    GPUInfo get_gpu_info(int dev_id);
    int num_gpus() const;
    int world_size() const {
        return world_size_;
    }

    void print_memory_summary();
    void freeze_model_memory();

    void device_foreach(std::function<void(int)> &fn);
    std::mutex log_mutex;
};

}

} // namespace bmengine::core
