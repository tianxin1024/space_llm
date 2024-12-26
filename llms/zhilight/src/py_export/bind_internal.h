#pragma once

#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <pybind11/pybind11.h>
// #include <torch/extension.h>

namespace bind {

namespace py = pybind11;

// layers
void define_layer_attention(py::module_ &m);
void define_layer_embedding(py::module_ &m);
void define_layer_feed_forward(py::module_ &m);
void define_layer_linear(py::module_ &m);
void define_layer_position_embedding(py::module_ &m);

void define_functions(py::module_ &m);

} // namespace bind
