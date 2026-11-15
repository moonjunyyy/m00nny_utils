#ifndef LIBM00NNY_H
#define LIBM00NNY_H

#include "common/common.h"
#include "common/macros.h"
#include "linalg/linalg.h"
#include "distance/distance.h"
#include "algorithm/algorithm.h"
#include "safetensor/safetensor.h"

static PyMethodDef M00NNY_Utils_Methods[] = {
    {"pairwise_distance",       (PyCFunction)dist_pairwise_distance,       METH_VARARGS | METH_KEYWORDS, "Computes the pairwise distance between two tensors."},
    {"compute_distance_matrix", (PyCFunction)dist_compute_distance_matrix, METH_VARARGS | METH_KEYWORDS, "Computes the distance matrix between two tensors."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef M00NNY_Utils_Module =
{
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "m00nny_utils",
    .m_doc  = "Custom Pytorch C++ Extension for Machine Learning",
    .m_size = -1,
    .m_methods = M00NNY_Utils_Methods,
};

// Submodules of m00nny_utils, added in this order by PyInit_m00nny_utils (m00nny.cpp).
// Each is a directory csrc/<name>/ whose <name>.cpp defines PyInit_<name>.
static const struct
{
    const char *name;
    PyObject *(*init)(void);
} M00NNY_Utils_Submodules[] =
{
    {"algorithm", PyInit_algorithm},
    {"linalg",    PyInit_linalg},
    {"safetensor", PyInit_safetensor},
};
#endif // LIBM00NNY_H
