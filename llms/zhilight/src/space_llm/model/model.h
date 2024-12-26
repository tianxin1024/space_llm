#pragma once
#include <core/core.h>
#include "model/model_config.hpp"
#include "model/model_util.h"

namespace model {
using namespace bmengine;

class ModelBase : public core::Layer {
public:
    ModelConfig cfg;
    std::string model_type;
    int num_layers;
    int dim_model;
    int num_heads;
    int dim_head;
    int dim_ff{0};
    int vocab_size;
    float eps;
    int num_kv_heads;
    core::DataType dtype;

    ModelBase(ModelConfig d) :
        cfg(d),
        model_type(d.model_type),
        num_layers(d.num_layers),
        dim_model(d.dim_model),
        num_heads(d.num_heads),
        dim_head(d.dim_head),
        dim_ff(d.dim_ff),
        vocab_size(d.vocab_size),
        eps(d.eps),
        num_kv_heads(d.num_kv_heads),
        dtype(d.dtype) {
    }

    ModelBase(const ModelBase &) = delete;
    ModelBase(ModelBase &&) = delete;
};

} // namespace model
