#include "utils/utils.h"

using namespace bmengine;

namespace utils {

void load_state_dict(
    bmengine::core::Context &ctx,
    const std::map<std::string, bmengine::core::Tensor> &state_dict,
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
} // namespace utils
