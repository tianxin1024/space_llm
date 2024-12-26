#pragma once
#include "core/core.h"
#include <memory>
#include <vector>

namespace kvcache {

using namespace bmengine;
using std::shared_ptr;

class KVCache {
protected:
    int batch_size;
    size_t num_layers, num_heads, dim_head;
    core::DataType dtype;
    std::vector<int> layer_devices;
    bool parallel;
    bool BSHD;
    bool continuous{false};

public:
    KVCache(
        int num_layers,
        int num_heads,
        int dim_head,
        core::DataType dtype,
        bool parallel = false,
        bool BSHD = false) :
        KVCache(-1, num_layers, num_heads, dim_head, dtype, parallel, BSHD) {
    }
    KVCache(
        int batch_size,
        int num_layers,
        int num_heads,
        int dim_head,
        core::DataType dtype,
        bool parallel,
        bool BSHD);

    virtual const core::Tensor &operator[](int i) const = 0;
    virtual core::Tensor &operator[](int i) = 0;
    virtual void resize(const core::Context &ctx, size_t new_length) = 0;

    void set_layer_devices(const std::vector<int> &layer_devices) {
        this->layer_devices = layer_devices;
    }
    const std::vector<int> &get_layer_devices() const {
        return layer_devices;
    }
    size_t get_num_layers() const {
        return num_layers;
    }
    bool is_BSHD() const {
        return BSHD;
    }
    void set_continuous(bool b) {
        continuous = b;
    }
};

struct KVCacheConfig {
    int num_layers;
    int num_heads;
    int dim_head;
    core::DataType dtype;
    bool BSHD;
    shared_ptr<core::DataType> scale_dtype; // optional
    std::vector<int> layer_devices;

    bool is_quant() const {
        return scale_dtype.get();
    }
};

} // namespace kvcache
