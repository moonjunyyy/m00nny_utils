#ifndef LIBM00NNY_DISTANCE_H
#define LIBM00NNY_DISTANCE_H

#include "common.h"
#include "macros.h"

namespace m00nny {

enum DistanceMetric { Cosine, Manhattan, Euclidean };
torch::Tensor pairwise_distance       (torch::Tensor& X, torch::Tensor& Y, int8_t metric);
torch::Tensor compute_distance_matrix (torch::Tensor& X, torch::Tensor& Y, int64_t batchsize, int8_t metric);

torch::Tensor& pairwise_distance_out      (torch::Tensor& Out, torch::Tensor& X, torch::Tensor& Y, int8_t metric);
torch::Tensor& compute_distance_matrix_out(torch::Tensor& Out, torch::Tensor& X, torch::Tensor& Y, int64_t batchsize, int8_t metric);

const char* metric_str(int8_t metric);

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