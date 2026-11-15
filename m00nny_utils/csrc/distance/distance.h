#ifndef LIBM00NNY_DISTANCE_H
#define LIBM00NNY_DISTANCE_H

#include "common/common.h"
#include "common/torch_common.h"
#include "common/macros.h"
#include "common/pyarg.h"

#include <utility>

namespace m00nny {

enum DistanceMetric { Cosine, Manhattan, Euclidean };

// The metric names. A non-negative integer p also names a metric: the Lp norm (Manhattan = 1, Euclidean = 2).
inline constexpr std::pair<const char*, int8_t> kMetricNames[] = {
    {"cosine", Cosine}, {"manhattan", Manhattan}, {"euclidean", Euclidean}};

torch::Tensor pairwise_distance       (torch::Tensor& X, torch::Tensor& Y, int8_t metric);
torch::Tensor compute_distance_matrix (torch::Tensor& X, torch::Tensor& Y, int64_t batchsize, int8_t metric);

torch::Tensor& pairwise_distance_out      (torch::Tensor& Out, torch::Tensor& X, torch::Tensor& Y, int8_t metric);
torch::Tensor& compute_distance_matrix_out(torch::Tensor& Out, torch::Tensor& X, torch::Tensor& Y, int64_t batchsize, int8_t metric);

const char* metric_str(int8_t metric);

// A metric argument of a Python call: one of kMetricNames or a non-negative int (see pyarg.h for the rule).
bool metric_arg(PyObject* o, const char* name, int8_t dflt, int8_t& out);

} // namespace m00nny

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

PyObject* dist_pairwise_distance       (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* dist_compute_distance_matrix (PyObject *self, PyObject *args, PyObject *kwds);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // LIBM00NNY_DISTANCE_H
