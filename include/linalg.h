#ifndef LIBM00NNY_LINALG_H
#define LIBM00NNY_LINALG_H

#include "common.h"
#include "macros.h"

namespace m00nny {
    torch::Tensor
    kronecker_product(torch::Tensor& X, torch::Tensor& Y);

    torch::Tensor&
    kronecker_product_out(torch::Tensor& Out, torch::Tensor& X, torch::Tensor& Y);
    
    torch::Tensor
    inverse(torch::Tensor& X);
    
    torch::Tensor
    orthogonalize(torch::Tensor& X);

    std::tuple<torch::Tensor, torch::Tensor>
    eigen_decomposition(torch::Tensor& X, int64_t lowrank, int64_t max_iters=2000, double threshold=1e-8);
    
    std::tuple<torch::Tensor, torch::Tensor>
    eigen_decomposition_new(torch::Tensor& X, int64_t lowrank, int64_t max_iters=2000, double threshold=1e-8);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    singular_value_decomposition(torch::Tensor& X, int64_t lowrank, int64_t max_iters=2000, double threshold=1e-8);
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#define PY_SSIZE_T_CLEAN

PyObject* linalg_kronecker_product (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_inverse (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_orthogonalize (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_eigen_decomposition (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_eigen_decomposition_new (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_singular_value_decomposition (PyObject *self, PyObject *args, PyObject *kwds);

#ifdef __cplusplus
}
#endif // __cplusplus

static PyMethodDef linalg_methods[] = {
    {"kronecker_product", (PyCFunction)linalg_kronecker_product, METH_VARARGS | METH_KEYWORDS, "Kronecker product of two matrices"},
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