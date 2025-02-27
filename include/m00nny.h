#ifndef LIBM00NNY_H
#define LIBM00NNY_H

#include "common.h"
#include "macros.h"
#include "linalg.h"
#include "distance.h"
#include "algorithm.h"

static PyMethodDef M00NNY_Utils_Methods[] = {
    {"pairwise_distance",       (PyCFunction)dist_pairwise_distance,       METH_VARARGS | METH_KEYWORDS, "Computes the pairwise distance between two tensors."},
    {"compute_distance_matrix", (PyCFunction)dist_compute_distance_matrix, METH_VARARGS | METH_KEYWORDS, "Computes the distance matrix between two tensors."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef M00NNY_Utils_Module =
{
    PyModuleDef_HEAD_INIT,
    .m_name = "m00nny_utils",
    .m_doc  = "Custom Pytorch C++ Extension for Machine Learning",
    .m_size = -1,
    .m_methods = M00NNY_Utils_Methods,
};
#endif // LIBM00NNY_H