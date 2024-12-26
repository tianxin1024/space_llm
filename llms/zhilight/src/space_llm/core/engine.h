#pragma once
#include "core/export.h"
#include <memory>
#include <vector>
#include <functional>

namespace bmengine {

namespace core {

class Context;
class MemoryAllocator;

struct DeviceConfiguration {
    int device_id;
    size_t memory_limit;

    DeviceConfiguration(int device_id, size_t memory_limit) :
        device_id(device_id), memory_limit(memory_limit) {
    }
};

struct GPUInfo {
    int real_device_idx;
    int compute_capability;
    size_t total_memory;
    size_t free_memory;
    size_t alloc_memory;
};

class EngineImpl;
// Engine can be accessed from multiple threads.
class BMENGINE_EXPORT Engine {
    friend class DistributedTensorImpl;
    std::unique_ptr<EngineImpl> pimpl;

public:
    Engine(const std::vector<DeviceConfiguration> &cfg, int tp = 0);
    ~Engine();

    Context create_context(const std::vector<int> &devices) const;
    Context create_context() const; // use all devices
    Context create_context_rank(int rank) const;
    int num_gpus() const;
    int world_size() const;
    GPUInfo get_gpu_info(int device_idx) const;

    // Disable copy
    Engine(const Engine &) = delete;
    Engine(Engine &&) = delete;

    void device_foreach(std::function<void(int)> fn);
    void print_memory_summary();
    void freeze_model_memory();
    MemoryAllocator *get_allocator(int dev_id);
};

}

} // namespace bmengine::core
