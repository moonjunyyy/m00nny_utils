#ifndef LIBM00NNY_LINALG_H
#define LIBM00NNY_LINALG_H

#include "common/torch_common.h"
#include "common/common.h"
#include "common/macros.h"
#include "common/pyarg.h"

namespace m00nny {
namespace internal {
    void check_square(const torch::Tensor& X, const std::string& func_name);
    // Symmetric up to `tolerance` times the input's unit roundoff, relative to the largest entry, per matrix.
    void check_symmetric(const torch::Tensor& X, double roundoff, const std::string& func_name);
}
    // Defaults shared by the C++ functions, their Python bindings and linalg/__init__.pyi.
    inline constexpr int64_t kEigenMaxIters  = 1000;   // warm-start plus Newton iterations
    inline constexpr double  kEigenThreshold = 1e-10;  // a pair is converged when |Xv - lv| <= threshold * |X|_F

    torch::Tensor
    inverse(const torch::Tensor& X_const);

    torch::Tensor
    orthogonalize(const torch::Tensor& X_const);

    std::tuple<torch::Tensor, torch::Tensor>
    eigen_decomposition(const torch::Tensor& X, int64_t lowrank = -1, int64_t max_iters = kEigenMaxIters, double threshold = kEigenThreshold);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    singular_value_decomposition(const torch::Tensor& X, int64_t lowrank = -1, int64_t max_iters = kEigenMaxIters, double threshold = kEigenThreshold);
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#define PY_SSIZE_T_CLEAN

PyObject* linalg_inverse (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_orthogonalize (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_eigen_decomposition (PyObject *self, PyObject *args, PyObject *kwds);
PyObject* linalg_singular_value_decomposition (PyObject *self, PyObject *args, PyObject *kwds);

#ifdef __cplusplus
}
#endif // __cplusplus

static PyMethodDef linalg_methods[] = {
    {"inverse", (PyCFunction)linalg_inverse, METH_VARARGS | METH_KEYWORDS, "Inverse of a square matrix (Gauss-Jordan with partial pivoting)"},
    {"orthogonalize", (PyCFunction)linalg_orthogonalize, METH_VARARGS | METH_KEYWORDS, "Orthonormalize the columns of a matrix (Gram-Schmidt, twice)"},
    {"eigen_decomposition", (PyCFunction)linalg_eigen_decomposition, METH_VARARGS | METH_KEYWORDS, "Eigen decomposition of a symmetric matrix (Newton on the Lagrangian)"},
    {"singular_value_decomposition", (PyCFunction)linalg_singular_value_decomposition, METH_VARARGS | METH_KEYWORDS, "Thin singular value decomposition"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef Linalg_Module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "linalg",
    .m_doc  = "Custom Pytorch C++ Extension for Linear Algebra",
    .m_size = -1,
    .m_methods = linalg_methods,
};

PyMODINIT_FUNC PyInit_linalg(void);

#endif // LIBM00NNY_LINALG_H
