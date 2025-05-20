#ifndef LIBM00NNY_LINALG_H
#define LIBM00NNY_LINALG_H

#include "common/torch_common.h"
#include "common/common.h"
#include "common/macros.h"
namespace m00nny {
namespace internal {
    void check_square(const torch::Tensor& X, const std::string& func_name);
    // void check_symmetric(const torch::Tensor& X, const std::string& func_name);
}

    torch::Tensor
    inverse(const torch::Tensor& X_const);
    
    torch::Tensor
    orthogonalize(const torch::Tensor& X_const);

    std::tuple<torch::Tensor, torch::Tensor>
    eigen_decomposition(const torch::Tensor& X, int64_t lowrank, int64_t max_iters=2000, double threshold=1e-6);
    
    std::tuple<torch::Tensor, torch::Tensor>
    eigen_decomposition_new(const torch::Tensor& X, int64_t lowrank, int64_t max_iters=2000, double threshold=1e-6, double damping_factor=1.0);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    singular_value_decomposition(const torch::Tensor& X, int64_t lowrank, int64_t max_iters=2000, double threshold=1e-6, double damping_factor=1.0);
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#define PY_SSIZE_T_CLEAN

PyObject* linalg_inverse (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_orthogonalize (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_eigen_decomposition (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_eigen_decomposition_new (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_singular_value_decomposition (PyObject *self, PyObject *args, PyObject *kwds);

#ifdef __cplusplus
}
#endif // __cplusplus

static PyMethodDef linalg_methods[] = {
    {"inverse", (PyCFunction)linalg_inverse, METH_VARARGS | METH_KEYWORDS, "Inverse of a square matrix"},
    {"orthogonalize", (PyCFunction)linalg_orthogonalize, METH_VARARGS | METH_KEYWORDS, "Orthogonalize a matrix"},
    {"eigen_decomposition", (PyCFunction)linalg_eigen_decomposition, METH_VARARGS | METH_KEYWORDS, "Eigen decomposition of a square matrix"},
    {"eigen_decomposition_new", (PyCFunction)linalg_eigen_decomposition_new, METH_VARARGS | METH_KEYWORDS, "Eigen decomposition of a square matrix"},
    {"singular_value_decomposition", (PyCFunction)linalg_singular_value_decomposition, METH_VARARGS | METH_KEYWORDS, "Singular value decomposition of a matrix"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef Linalg_Module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "linalg",
    .m_doc  = "Custom Pytorch C++ Extension for Linear Algebra",
    .m_size = -1,
    .m_methods = linalg_methods,
};

#endif // LIBM00NNY_LINALG_H