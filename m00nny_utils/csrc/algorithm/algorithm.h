#ifndef LIBM00NNY_ALGORITHM_H
#define LIBM00NNY_ALGORITHM_H

#include "common/common.h"
#include "common/macros.h"
#include "algorithm/tsne.h"
#include "algorithm/kmeans.h"

static PyModuleDef Algorithm_Module =
{
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "algorithm",
    .m_doc  = "Custom Pytorch C++ Extension for Machine Learning Algorithms",
    .m_size = -1
};

// Types of m00nny_utils.algorithm, added in this order by PyInit_algorithm (algorithm.cpp).
static PyTypeObject* const Algorithm_Types[] =
{
    &TSNEType,
    &KMeansType,
};

PyMODINIT_FUNC PyInit_algorithm(void);

#endif // LIBM00NNY_ALGORITHM_H
