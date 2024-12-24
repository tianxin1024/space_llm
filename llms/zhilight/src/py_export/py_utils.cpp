#include "py_export/bind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_fp16.h>
#include <sstream>
#include <iostream>
#include "core/tensor.h"
#include "model/model.h"

namespace bind {

using bmengine::core::Context;
using bmengine::core::DataType;
using bmengine::core::Tensor;

uint16_t float2half(float x) {
    __half_raw v = __float2half(x);
    return v.x;
}

float half2float(uint16_t x) {
    __half_raw v;
    v.x = x;
    return __half2float(v);
}

// Create a tensor reference to numpy array's underlying data.
// No data coping happens.
const Tensor numpy_to_tensor(const std::string &name, const py::array &arr) {
    py::format_descriptor<float>::format();
    py::buffer_info buf = arr.request();
    auto dtype = numpy_dtype_to_bmengine(arr.dtype());
    const auto tensor = Tensor::from_external(
        *reinterpret_cast<std::vector<size_t> *>(&buf.shape),
        dtype,
        buf.ptr,
        arr.nbytes());
    tensor.set_name(name);
    return std::move(tensor);
}

std::map<std::string, const Tensor> numpy_to_tensor(
    const std::map<std::string, py::array> &state_dict) {
    std::map<std::string, const Tensor> tensor_dict;
    for (auto &it : state_dict) {
        tensor_dict.emplace(it.first, std::move(bind::numpy_to_tensor(it.first, it.second)));
    }
    return tensor_dict;
}

std::map<std::string, const bmengine::core::Tensor> numpy_to_tensor(py::dict state_dict) {
    std::map<std::string, const Tensor> tensor_dict;
    for (auto &it : state_dict) {
        std::string name = py::str(it.first);
        tensor_dict.emplace(name, std::move(bind::numpy_to_tensor(name, reinterpret_cast<const py::array &>(it.second))));
    }
    return tensor_dict;
}

// load recursively
void load_state_dict_new(
    bmengine::core::Context &ctx,
    const std::map<std::string, py::array> &state_dict,
    const std::string &prefix,
    bmengine::core::Layer *model) {
    std::map<std::string, const Tensor> tensor_dict = numpy_to_tensor(state_dict);

    // load params recursively
    model->load_state_dict(ctx, tensor_dict, prefix);

    // check all params was initialized
    auto named_parameters = model->named_parameters(prefix);
    for (auto &p : named_parameters) {
        if (p.second->data() == nullptr) {
            throw std::runtime_error("state_dict missing: " + p.first);
        }
    }
}

void load_state_dict(
    bmengine::core::Context &ctx,
    const std::map<std::string, py::array> &state_dict,
    std::map<const std::string, bmengine::core::Tensor *> named_params,
    bool parallel) {
    for (auto it : named_params) {
        BM_ASSERT(it.second, it.first + std::string(" not in named_params"));
        auto p = state_dict.find(it.first);
        if (p != state_dict.end()) {
            if (!parallel || ctx.rank() == 0) {
                auto buf = p->second.request();
                BM_ASSERT(
                    it.second->ndim() == buf.ndim,
                    it.first + " ndim miss match: " + std::to_string(it.second->ndim())
                        + " != " + std::to_string(buf.ndim));
                for (int i = 0; i < it.second->ndim(); ++i) {
                    std::stringstream ss;
                    ss << "model[" << i << "]=" << it.second->shape()[i] << ", state[" << i
                       << "]=" << buf.shape[i];
                    // std::cout << ss.str() + "=>wjj" << std::endl;
                    BM_ASSERT(
                        it.second->shape()[i] == buf.shape[i],
                        "Parameter `" + it.first + "` has different shape" + ss.str());
                }
                BM_ASSERT(
                    it.second->nbytes() == (buf.size * buf.itemsize),
                    it.first + " size miss match: " + std::to_string(it.second->nbytes())
                        + " != " + std::to_string(buf.size * buf.itemsize));
                // TODO add dtype check with numpy.
                // BM_ASSERT(py::dtype == p->second.dtype().num(), it.first + " dtype miss match" +
                // std::string(get_data_type_name(it.second->dtype())) + p->second.dtype().char_());
                ctx.init_parameter(it.first, it.second);
                it.second->from_buffer(buf.ptr);
            } else {
                it.second->from_buffer(nullptr);
            }

        } else {
            std::stringstream ss;
            ss << "state_dict missing: " << it.first;
            throw std::runtime_error(ss.str());
        }
    }
}

void convert_results(const std::vector<generator::SearchResults> &results_vec, py::list *res) {
    for (const auto &results : results_vec) {
        const generator::SearchResult &result = results.results[0];
        res->append(py::make_tuple(
            py::cast(result.tokens),
            py::cast(result.logprobs),
            py::cast(result.cumulative_logprob),
            py::cast(result.score)));
    }
}

void convert_multi_results(
    const std::vector<generator::SearchResults> &results_vec, py::list *res) {
    for (const auto &results : results_vec) {
        py::list l;
        for (auto result : results.results) {
            l.append(py::make_tuple(
                py::cast(result.tokens),
                py::cast(result.logprobs),
                py::cast(result.cumulative_logprob),
                py::cast(result.score)));
        }
        res->append(l);
    }
}

template <typename T>
std::vector<T> to_1d_vector(const py::list &z) {
    std::vector<T> v;
    for (const auto &it : z) {
        v.emplace_back(it.cast<T>());
    }
    return v;
}

std::vector<int> to_int_vector(const py::list &data_list) {
    return to_1d_vector<int>(data_list);
}

std::vector<std::string> to_string_vector(const py::list &data_list) {
    return to_1d_vector<std::string>(data_list);
}

template <typename T>
std::vector<std::vector<T>> to_2d_vector(const py::list &data_list) {
    std::vector<std::vector<T>> c_data_list;
    for (const auto &it : data_list) {
        std::vector<T> token_ids;
        for (const auto &jt : it) {
            token_ids.emplace_back(jt.cast<T>());
        }
        c_data_list.emplace_back(token_ids);
    }
    return c_data_list;
}

std::vector<std::vector<int>> to_2d_int_vector(const py::list &data_list) {
    return to_2d_vector<int>(data_list);
}

std::vector<std::vector<bool>> to_2d_bool_vector(const py::list &data_list) {
    return to_2d_vector<bool>(data_list);
}

bmengine::core::DataType numpy_dtype_to_bmengine(py::dtype dtype) {
    switch (dtype.char_()) {
    case 'd': return bmengine::core::DataType::kDouble;
    case 'f': return bmengine::core::DataType::kFloat;
    case 'e': return bmengine::core::DataType::kHalf;
    case 'b': return bmengine::core::DataType::kInt8;
    case 'h': return bmengine::core::DataType::kInt16;
    case 'i': return bmengine::core::DataType::kInt32;
    default: break;
    }
    throw std::runtime_error(
        std::string("can't convert np.ndarray of type ") + dtype.char_()
        + "The only supported types are: "
          "float63, float32, float16, int32, int16, half and int8.");
}

template <typename T>
static void set_attr(const py::dict &cfg, const char *name, T &attr) {
    if (cfg.template contains(name) && !py::isinstance<py::none>(cfg[name]))
        attr = cfg[name].cast<T>();
}

void pydict_to_rope_config(const py::dict &d, model::RopeConfig &config) {
    set_attr(d, "type", config.type);
    set_attr(d, "factor", config.factor);
    set_attr(d, "attn_factor", config.attn_factor);
    set_attr(d, "beta_fast", config.beta_fast);
    set_attr(d, "beta_slow", config.beta_slow);
    set_attr(d, "mscale", config.mscale);
    set_attr(d, "mscale_all_dim", config.mscale_all_dim);
    set_attr(d, "original_max_position_embeddings", config.original_max_position);
}

model::ModelConfig pydict_to_model_config(py::dict &cfg) {
    std::string model_type = cfg.contains("model_type") ? cfg["model_type"].cast<std::string>() : "cpm_caterpillar";
    int num_layers = cfg["num_layers"].cast<int>();
    int dim_model = cfg["dim_model"].cast<int>();
    int num_heads = cfg["num_heads"].cast<int>();
    int dim_head = cfg["dim_head"].cast<int>();
    int dim_ff = cfg["dim_ff"].cast<int>();
    int vocab_size = cfg["vocab_size"].cast<int>();
    float eps = cfg.contains("eps") ? cfg["eps"].cast<float>() : 1e-5;
    int num_kv_heads = cfg.contains("num_kv_heads") ? cfg["num_kv_heads"].cast<int>() : -1;
    auto mask_modules = cfg.contains("mask_modules") ? bind::to_2d_bool_vector(cfg["mask_modules"]) : std::vector<std::vector<bool>>{};
    bool scale_weights = cfg.contains("scale_weights") ? cfg["scale_weights"].cast<bool>() : false;
    bool weight_transposed =
        cfg.contains("weight_transposed") ? cfg["weight_transposed"].cast<bool>() : false;

    int dim_model_base =
        cfg.contains("dim_model_base") ? cfg["dim_model_base"].cast<int>() : 0;
    float scale_depth =
        cfg.contains("scale_depth") ? cfg["scale_depth"].cast<float>() : 1.0;
    float scale_emb =
        cfg.contains("scale_emb") ? cfg["scale_emb"].cast<float>() : 1.0;
    auto dtype = cfg.contains("dtype") ? cfg["dtype"].cast<const std::string>() : "half";
    auto data_type = bmengine::core::name_to_data_type(dtype);
    if (cfg.contains("bf16") && cfg["bf16"].cast<bool>() || cfg.contains("bfloat16") && cfg["bfloat16"].cast<bool>() || cfg.contains("bfloat") && cfg["bfloat"].cast<bool>())
        data_type = bmengine::core::DataType::kBFloat16;
    if (cfg.contains("force_half") && cfg["force_half"].cast<bool>())
        data_type = bmengine::core::DataType::kHalf;

    model::ModelConfig config{model_type, num_layers, dim_model, num_heads, dim_head, dim_ff, vocab_size,
                              eps, num_kv_heads, mask_modules, scale_weights, weight_transposed, dim_model_base, scale_depth, scale_emb, data_type};

    config.pos_bias_type =
        cfg.contains("pos_bias_type") ? cfg["pos_bias_type"].cast<std::string>() : "rotary";
    config.activate_fn =
        cfg.contains("activate_fn") ? cfg["activate_fn"].cast<std::string>() : "silu";
    BM_ASSERT(config.pos_bias_type == "rotary" || config.pos_bias_type == "relative",
              "Unsupported pos_bias_type");
    BM_ASSERT(config.activate_fn == "silu" || config.activate_fn == "gelu",
              "Unsupported activate_fn");

    config.tie_lm_head = cfg.contains("tie_lm_head") && cfg["tie_lm_head"].cast<bool>();

    if (cfg.contains("rope_theta"))
        config.rope_theta = cfg["rope_theta"].cast<float>();

    if (cfg.contains("rope_scaling") && py::isinstance<py::dict>(cfg["rope_scaling"])) {
        pydict_to_rope_config(cfg["rope_scaling"], config.rope_cfg);
    }
    if (cfg.contains("max_position_embeddings"))
        config.max_position_embeddings = cfg["max_position_embeddings"].cast<int>();

    // moe config
    set_attr(cfg, "moe_num_experts", config.moe_num_experts);
    set_attr(cfg, "num_local_experts", config.moe_num_experts);
    set_attr(cfg, "num_experts", config.moe_num_experts);
    set_attr(cfg, "n_routed_experts", config.moe_num_experts);
    set_attr(cfg, "moe_top_k", config.moe_top_k);
    set_attr(cfg, "num_experts_per_tok", config.moe_top_k);
    set_attr(cfg, "moe_intermediate_size", config.moe_intermediate_size);
    set_attr(cfg, "shared_expert_intermediate_size", config.shared_expert_intermediate_size);
    set_attr(cfg, "norm_topk_prob", config.norm_topk_prob);
    set_attr(cfg, "first_k_dense_replace", config.first_k_dense_replace);
    set_attr(cfg, "routed_scaling_factor", config.routed_scaling_factor);
    // MOE of DeepSeek
    set_attr(cfg, "n_group", config.moe_n_group);
    set_attr(cfg, "topk_group", config.moe_topk_group);

    // MLA config
    set_attr(cfg, "q_lora_rank", config.q_lora_rank);
    set_attr(cfg, "kv_lora_rank", config.kv_lora_rank);
    set_attr(cfg, "qk_nope_head_dim", config.qk_nope_head_dim);
    set_attr(cfg, "qk_rope_head_dim", config.qk_rope_head_dim);
    set_attr(cfg, "v_head_dim", config.v_head_dim);

    set_attr(cfg, "use_qk_norm", config.use_qk_norm);
    set_attr(cfg, "logit_scale", config.logit_scale);

    return config;
}
} // namespace bind
