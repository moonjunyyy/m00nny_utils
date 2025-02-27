#ifndef LIBM00NNY_ALGORITHM_H
#define LIBM00NNY_ALGORITHM_H

#include "common.h"
#include "macros.h"
#include "tsne.h"
#include "kmeans.h"

static PyModuleDef Algorithm_Module =
{
    PyModuleDef_HEAD_INIT,
    .m_name = "algorithm",
    .m_doc  = "Custom Pytorch C++ Extension for Machine Learning Algorithms",
    .m_size = -1
};

#endif // LIBM00NNY_ALGORITHM_H