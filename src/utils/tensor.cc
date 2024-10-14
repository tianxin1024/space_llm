#include "utils/tensor.h"

namespace space_llm {

Tensor::Tensor() :
    // a none tensor.
    where(MEMORY_CPU),
    type(TYPE_INVALID),
    shape({}),
    data(nullptr),
    offsets({}) { // only a record to record offset
}

Tensor::Tensor(const MemoryType _where, const DataType _type, const std::vector<size_t> _shape, const void *_data) :
    where(_where), type(_type), shape(_shape), data(_data) {
}

Tensor::Tensor(const MemoryType _where,
               const DataType _type,
               const std::vector<size_t> _shape,
               const void *_data,
               const std::vector<size_t> _offset) :
    where(_where),
    type(_type), shape(_shape), data(_data), offsets(_offset) {
}

size_t Tensor::size() const {
    if (data == nullptr || shape.size() == 0) {
        return 0;
    }
    return std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
}

size_t Tensor::sizeBytes() const {
    return size() * Tensor::getTypeSize(type);
}

std::string Tensor::getNumpyTypeDesc(DataType type) const {
    static const std::unordered_map<DataType, std::string> type_map{{TYPE_INVALID, "x"},
                                                                    {TYPE_BOOL, "?"},
                                                                    {TYPE_BYTES, "b"},
                                                                    {TYPE_UINT8, "u1"},
                                                                    {TYPE_UINT16, "u2"},
                                                                    {TYPE_UINT32, "u4"},
                                                                    {TYPE_UINT64, "u8"},
                                                                    {TYPE_INT8, "i1"},
                                                                    {TYPE_INT16, "i2"},
                                                                    {TYPE_INT32, "i4"},
                                                                    {TYPE_INT64, "i8"},
                                                                    {TYPE_FP16, "f2"},
                                                                    {TYPE_FP32, "f4"},
                                                                    {TYPE_FP64, "f8"}};

    if (type == TYPE_BF16) {
        QK_LOG_WARNING("getNumpyTypeDesc(TYPE_BF16) returns an invalid type 'x' since Numpy doesn't "
                       "support bfloat16 as of now, it will be properly extended if numpy supports. "
                       "Please refer for the discussions https://github.com/numpy/numpy/issues/19808.");
    }

    return type_map.count(type) > 0 ? type_map.at(type) : "x";
}

DataType Tensor::typeFromNumpyDesc(std::string type) {
    static const std::unordered_map<std::string, DataType> type_map{{"?", TYPE_BOOL},
                                                                    {"b", TYPE_BYTES},
                                                                    {"u1", TYPE_UINT8},
                                                                    {"u2", TYPE_UINT16},
                                                                    {"u4", TYPE_UINT32},
                                                                    {"u8", TYPE_UINT64},
                                                                    {"i1", TYPE_INT8},
                                                                    {"i2", TYPE_INT16},
                                                                    {"i4", TYPE_INT32},
                                                                    {"i8", TYPE_INT64},
                                                                    {"f2", TYPE_FP16},
                                                                    {"f4", TYPE_FP32},
                                                                    {"f8", TYPE_FP64}};
    return type_map.at(type);
}

size_t Tensor::getTypeSize(DataType type) {
    static const std::unordered_map<DataType, size_t> type_map{{TYPE_BOOL, sizeof(bool)},
                                                               {TYPE_BYTES, sizeof(char)},
                                                               {TYPE_UINT8, sizeof(uint8_t)},
                                                               {TYPE_UINT16, sizeof(uint16_t)},
                                                               {TYPE_UINT32, sizeof(uint32_t)},
                                                               {TYPE_UINT64, sizeof(uint64_t)},
                                                               {TYPE_INT8, sizeof(int8_t)},
                                                               {TYPE_INT16, sizeof(int16_t)},
                                                               {TYPE_INT32, sizeof(int32_t)},
                                                               {TYPE_INT64, sizeof(int64_t)},
#ifdef ENABLE_BF16
                                                               {TYPE_BF16, sizeof(__nv_bfloat16)},
#endif
#ifdef ENABLE_FP8
                                                               {TYPE_FP8_E4M3, sizeof(__nv_fp8_e4m3)},
#endif
                                                               {TYPE_FP16, sizeof(half)},
                                                               {TYPE_FP32, sizeof(float)},
                                                               {TYPE_FP64, sizeof(double)}};
    return type_map.at(type);
}

Tensor Tensor::slice(std::vector<size_t> shape, size_t offset) const {
    if (this->data != nullptr) {
        size_t n_elts = this->size();
        size_t n_sliced_elts = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        QK_CHECK_WITH_INFO(
            n_sliced_elts + offset <= n_elts,
            fmtstr("The number (%ld) of elements of sliced tensor exceeds that (%ld) of the original tensor",
                   n_sliced_elts + offset,
                   n_elts));
    }
    return Tensor(this->where, this->type, shape, this->getPtrWithOffset(offset));
}

TensorMap::TensorMap(const std::unordered_map<std::string, Tensor> &tensor_map) {
    for (auto &kv : tensor_map) {
        if (isValid(kv.second)) {
            insert(kv.first, kv.second);
        } else {
            QK_LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", kv.first.c_str()));
        }
    }
}

TensorMap::TensorMap(const std::vector<Tensor> &tensor_map) {
    for (size_t i = 0; i < tensor_map.size(); i++) {
        insert(std::to_string(i), tensor_map[i]);
    }
}

TensorMap::TensorMap(std::initializer_list<std::pair<std::string, Tensor>> tensor_map) {
    for (auto &pair : tensor_map) {
        if (isValid(pair.second)) {
            insert(pair.first, pair.second);
        } else {
            QK_LOG_DEBUG(fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
        }
    }
}

TensorMap::~TensorMap() {
    tensor_map_.clear();
}

std::vector<std::string> TensorMap::keys() const {
    std::vector<std::string> key_names;
    for (auto &kv : tensor_map_) {
        key_names.push_back(kv.first);
    }
    return key_names;
}

} // namespace space_llm
