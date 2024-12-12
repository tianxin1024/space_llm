#include "core/tensor.h"
#include "private/memory.h"
#include "private/tensor_impl.h"
#include "core/exception.h"
#include <map>
#include <sstream>

namespace bmengine {
namespace core {

size_t get_elem_size(DataType dtype) {
    switch (dtype) {
    case DataType::kDouble: return sizeof(double);
    case DataType::kFloat: return sizeof(float);
    case DataType::kHalf: return sizeof(int16_t);
    case DataType::kBFloat16: return sizeof(int16_t);
    case DataType::kInt8: return sizeof(int8_t);
    case DataType::kInt16: return sizeof(int16_t);
    case DataType::kInt32: return sizeof(int32_t);
    default: return 1;
    }
}

size_t get_numel(std::initializer_list<int> size) {
    size_t num = 1;
    for (auto s : size) {
        num *= s;
    }
    return size.size() == 0 ? 0 : num;
}

size_t get_numel(const std::vector<size_t> &size) {
    size_t num = 1;
    for (auto s : size) {
        num *= s;
    }
    return size.empty() ? 0 : num;
}

using std::to_string;
void check_no_zero(const std::vector<size_t> &size) {
    for (size_t i = 0; i < size.size(); i++) {
        BM_ASSERT(size[i] > 0, "Invalid tensor size[" + to_string(i) + "]=" + to_string(size[i]));
    }
}

int TensorImpl::normalize_dim(int dim) const {
    dim = dim < 0 ? ndim() + dim : dim;
    BM_ASSERT(dim >= 0 && dim < ndim(), "Dim out of range: " + to_string(dim) + ">=" + to_string(ndim()));
    return dim;
}

size_t TensorImpl::size(int dim) const {
    return shape_[normalize_dim(dim)];
}

size_t TensorImpl::stride(int dim) const {
    return strides[normalize_dim(dim)];
}

size_t TensorImpl::stride_bytes(int dim) const {
    return stride(dim) * get_elem_size(dtype_);
}

void *TensorImpl::data() const {
    if (mem && mem->ptr != nullptr) {
        return (uint8_t *)mem->ptr + offset;
    } else {
        return nullptr;
    }
}
void *TensorImpl::nullable_data() const {
    if (mem && mem->ptr != nullptr) {
        return (uint8_t *)mem->ptr + offset;
    } else {
        return nullptr;
    }
}
void TensorImpl::check_mem() const {
    BM_ASSERT(mem != nullptr, "Tensor is empty");
}
void *TensorImpl::mutable_data() {
    check_mem();
    if (mem->ptr != nullptr) {
        return (uint8_t *)mem->ptr + offset;
    } else {
        return nullptr;
    }
}

size_t TensorImpl::mem_bytes() const {
    check_mem();
    return mem->ptr != nullptr ? mem->num_bytes : 0;
}

int TensorImpl::device() const {
    check_mem();
    return mem->dev;
}

std::unique_ptr<TensorImpl> TensorImpl::view_type(const std::vector<size_t> &size, DataType dtype, bool check_size) const {
    // TODO: check continuous
    BM_ASSERT(numel() > 0, "Tensor is empty");
    size_t total_size = 1;
    for (auto s : size) {
        total_size *= s;
    }
    if (check_size) {
        BM_ASSERT_EQ(total_size * get_elem_size(dtype), nbytes_, "Tensor size mismatch");
    }
    return std::move(std::make_unique<TensorImpl>(size, mem, offset, dtype));
}

std::unique_ptr<TensorImpl> TensorImpl::view(const std::vector<size_t> &size) const {
    return view_type(size, dtype_);
}

std::unique_ptr<TensorImpl> TensorImpl::view_unchecked(const std::vector<size_t> &size, DataType dtype) const {
    return view_type(size, dtype, false);
}

std::vector<std::unique_ptr<TensorImpl>> TensorImpl::chunk() const {
    std::vector<std::unique_ptr<TensorImpl>> chunks;
    BM_ASSERT(numel() > 0, "Tensor is empty");
    BM_ASSERT(ndim() > 1, "Tensor must be 2D or larger");
    auto new_size = size();
    new_size.erase(new_size.begin());
    size_t n_offset = offset;
    for (int i = 0; i < shape_[0]; ++i) {
        BM_ASSERT(
            (n_offset - offset + stride_bytes(0)) <= nbytes(),
            "chunk overflow:" + std::to_string(n_offset - offset) + "/"
                + std::to_string(stride_bytes(0)) + "/" + std::to_string(nbytes()));
        chunks.emplace_back(std::make_unique<TensorImpl>(new_size, mem, n_offset, dtype_));
        n_offset += stride_bytes(0);
    }
    return chunks;
}

std::unique_ptr<TensorImpl> TensorImpl::index_dim0(size_t i) const {
    auto new_shape = std::vector<size_t>(shape_.begin() + 1, shape_.end());
    return std::make_unique<TensorImpl>(
        new_shape, mem, offset + stride_bytes(0) * i, dtype_);
}

std::unique_ptr<TensorImpl> TensorImpl::slice_dim0(size_t from, size_t to) const {
    auto new_size = size();
    new_size[0] = to - from;
    return std::make_unique<TensorImpl>(
        new_size, mem, offset + stride_bytes(0) * from, dtype_);
}
std::unique_ptr<TensorImpl> TensorImpl::virtual_slice(
    size_t from, size_t len, int dim) const {
    dim = normalize_dim(dim);
    BM_ASSERT_EQ(dim, ndim() - 1, "Support only last dim");
    BM_ASSERT_LE(from + len, size(dim), "Out of ramge");
    auto new_size = size();
    new_size[dim] = len;
    auto ret = std::make_unique<TensorImpl>(
        new_size, mem, offset + stride_bytes(dim) * from, dtype_);
    ret->strides = this->strides; // use same storage
    return std::move(ret);
}

std::string TensorImpl::info(int level) const {
    std::ostringstream os;

    os << "Tensor(";
    if (level > 0)
        os << "id=" << id_ << ", name='" << name_ << "', addr=" << (void *)(this) << ", ";
    os << "shape=[";
    bool first_dim = true;
    for (auto s : shape_) {
        if (first_dim) {
            first_dim = false;
        } else {
            os << ", ";
        }
        os << s;
    }
    os << "], device=" << mem->dev << ", dtype=" << get_data_type_name(dtype_);
    if (level > 0)
        os << ", nElements=" << numel_ << ", nBytes=" << nbytes_;
    os << ")";
    return os.str();
}

DataType TensorImpl::dtype() const {
    return dtype_;
}

const std::string &TensorImpl::name() const {
    return name_;
}
void TensorImpl::set_name(const std::string &name) {
    this->name_ = name;
}

TensorImpl::TensorImpl(
    const std::vector<size_t> &shape, Memory mem, size_t offset, DataType dtype) :
    dtype_(dtype),
    shape_(shape), mem(mem), offset(offset) {
    numel_ = get_numel(shape);
    nbytes_ = numel_ * get_elem_size(dtype);

    size_t all_strides = numel_;
    for (size_t i = 0; i < shape_.size(); ++i) {
        all_strides /= shape_[i];
        strides.push_back(all_strides);
    }
}

}
} // namespace bmengine::core
