#pragma once

#include "utils/logger.h"
#include "utils/cuda_utils.h"

#include "stdlib.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <dirent.h>
#include <numeric>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace space_llm {

typedef enum datatype_enum {
    TYPE_INVALID,
    TYPE_BOOL,
    TYPE_UINT8,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_INT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FP16,
    TYPE_FP32,
    TYPE_FP64,
    TYPE_BYTES,
    TYPE_BF16,
    TYPE_FP8_E4M3,
    TYPE_STR,
    TYPE_VOID,
} DataType;

template <typename T>
DataType getTensorType() {
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return TYPE_FP32;
    } else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return TYPE_FP16;
    } else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return TYPE_INT32;
    } else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return TYPE_INT8;
    } else if (std::is_same<T, uint>::value || std::is_same<T, const uint>::value) {
        return TYPE_UINT32;
    } else if (std::is_same<T, unsigned long long int>::value || std::is_same<T, const unsigned long long int>::value) {
        return TYPE_UINT64;
    } else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return TYPE_BOOL;
    } else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return TYPE_BYTES;
    } else {
        return TYPE_INVALID;
    }
}

typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

struct Tensor {
    const MemoryType where;
    const DataType type;
    const std::vector<size_t> shape;
    const void *data; // TODO(bhseuh) modify from const void* to void* const
    const std::vector<size_t> offsets = std::vector<size_t>{};

    Tensor();
    Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void *_data);
    Tensor(const MemoryType _where,
           const DataType _type,
           const std::vector<size_t> _shape,
           const void *_data,
           const std::vector<size_t> _offset);

    size_t size() const;
    size_t sizeBytes() const;

    std::string getNumpyTypeDesc(DataType type) const;

    static DataType typeFromNumpyDesc(std::string type);
    static size_t getTypeSize(DataType type);

    template <typename T>
    inline T getVal(size_t index) const {
        QK_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        QK_CHECK(where == MEMORY_CPU);
        QK_CHECK(data != nullptr);
        QK_CHECK_WITH_INFO(index < size(), "index is larger than buffer size");

        if (getTensorType<T>() != type) {
            QK_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return ((T *)data)[index];
    }

    template <typename T>
    inline T getVal() const {
        QK_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            QK_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return getVal<T>(0);
    }

    template <typename T>
    inline T *getPtr() const {
        QK_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            QK_LOG_DEBUG("getPtr with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        return (T *)data;
    }

    inline void *getPtrWithOffset(size_t offset) const {
        QK_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (data == nullptr) {
            return (void *)data;
        } else {
            QK_CHECK_WITH_INFO(offset < size(), "offset is larger than buffer size");
            return (void *)((char *)data + offset * Tensor::getTypeSize(type));
        }
    }

    template <typename T>
    inline T *getPtrWithOffset(size_t offset) const {
        QK_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
        if (getTensorType<T>() != type) {
            QK_LOG_DEBUG("getVal with type %s, but data type is: %s",
                         getNumpyTypeDesc(getTensorType<T>()).c_str(),
                         getNumpyTypeDesc(type).c_str());
        }
        if (data == nullptr) {
            return (T *)data;
        } else {
            QK_CHECK_WITH_INFO(offset < size(),
                               fmtstr("offset (%lu) is larger than buffer size (%lu)", offset, size()));
            return ((T *)data) + offset;
        }
    }

}; // struct Tensor

class TensorMap {
private:
    std::unordered_map<std::string, Tensor> tensor_map_;

    inline bool isValid(const Tensor &tensor) {
        return tensor.size() > 0 && tensor.data != nullptr;
    }

public:
    TensorMap() = default;
    TensorMap(const std::unordered_map<std::string, Tensor> &tensor_map);
    TensorMap(const std::vector<Tensor> &tensor_map);
    TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map);
    ~TensorMap();

    inline size_t size() const {
        return tensor_map_.size();
    }

    inline bool isExist(const std::string &key) const {
        QK_LOG_DEBUG("%s for key: %s", __PRETTY_FUNCTION__, key.c_str());
        return tensor_map_.find(key) != tensor_map_.end();
    }

    std::vector<std::string> keys() const;

    inline void insert(const std::string &key, const Tensor &value) {
        QK_CHECK_WITH_INFO(!isExist(key), fmtstr("Duplicated key %s", key.c_str()));
        QK_CHECK_WITH_INFO(isValid(value), fmtstr("A none tensor or nullptr is not allowed (key is %s)", key.c_str()));
        tensor_map_.insert({key, value});
    }

    inline void insertIfValid(const std::string &key, const Tensor &value) {
        if (isValid(value)) {
            insert({key, value});
        }
    }

    inline void insert(std::pair<std::string, Tensor> p) {
        tensor_map_.insert(p);
    }

    // prevent converting int or size_t to string automatically
    Tensor at(int tmp) = delete;
    Tensor at(size_t tmp) = delete;

    inline Tensor &at(const std::string &key) {
        QK_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        QK_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor at(const std::string &key) const {
        QK_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }

    inline Tensor &at(const std::string &key, Tensor &default_tensor) {
        QK_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor at(const std::string &key, Tensor &default_tensor) const {
        QK_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor &at(const std::string &key, Tensor &&default_tensor) {
        QK_LOG_DEBUG("%s for key %s", __PRETTY_FUNCTION__, key.c_str());
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    inline Tensor at(const std::string &key, Tensor &&default_tensor) const {
        if (isExist(key)) {
            return tensor_map_.at(key);
        }
        return default_tensor;
    }

    template <typename T>
    inline T *getPtr(const std::string &key) const {
        QK_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key).getPtr<T>();
    }

    template <typename T>
    inline T *getPtr(const std::string &key, T *default_ptr) const {
        if (isExist(key)) {
            return tensor_map_.at(key).getPtr<T>();
        }
        return default_ptr;
    }

    template <typename T>
    inline T *getPtrWithOffset(const std::string &key, size_t index) const {
        QK_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key).getPtrWithOffset<T>(index);
    }

    template <typename T>
    inline T *getPtrWithOffset(const std::string &key, size_t index, T *default_ptr) const {
        if (isExist(key)) {
            return tensor_map_.at(key).getPtrWithOffset<T>(index);
        }
        return default_ptr;
    }

    template <typename T>
    inline T getVal(const std::string &key) const {
        QK_CHECK_WITH_INFO(isExist(key),
                           fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key).getVal<T>();
    }

    template <typename T>
    inline T getVal(const std::string &key, T default_value) const {
        if (isExist(key)) {
            return tensor_map_.at(key).getVal<T>();
        }
        return default_value;
    }
}; // class TensorMap

} // namespace space_llm
