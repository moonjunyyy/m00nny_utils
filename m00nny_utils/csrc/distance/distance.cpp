#include "distance/distance.h"

#include <cstdio>

torch::Tensor
m00nny::pairwise_distance
(torch::Tensor& X, torch::Tensor& Y, int8_t metric)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: m00nny::pairwise_distance\n")
    torch::Tensor _dist;
    if (metric < 0) throw std::invalid_argument("Invalid metric");
    else if (metric == Cosine)
    {
        _dist = 1 - torch::clamp(torch::mm(X, Y.t()) / (torch::norm(X, 2, 1, true) * torch::norm(Y, 2, 1, true).t() + 1e-8), -1.0, 1.0);
        return _dist;
    }
    _dist = torch::norm(X.unsqueeze(1) - Y.unsqueeze(0), metric, 2);
    return _dist;
}

torch::Tensor
m00nny::compute_distance_matrix
(torch::Tensor& X, torch::Tensor& Y, int64_t batchsize, int8_t metric)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: m00nny::compute_distance_matrix\n")
    if (batchsize <= 0) throw std::invalid_argument("compute_distance_matrix: batchsize must be positive");
    int64_t _x_size = X.size(0), _y_size = Y.size(0);
    torch::Tensor Out = torch::empty({_x_size, _y_size}, X.options());
    torch::Tensor _batched_Out;
    torch::Tensor _batched_X;
    for (int64_t i = 0; i < _x_size; i += batchsize)
    {
        _batched_Out = Out.slice(0, i, i+batchsize, 1);
        _batched_X   =   X.slice(0, i, i+batchsize, 1);
        pairwise_distance_out(_batched_Out, _batched_X, Y, metric);
    }
    return Out;
}

torch::Tensor&
m00nny::pairwise_distance_out
(torch::Tensor& Out, torch::Tensor& X, torch::Tensor& Y, int8_t metric)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: m00nny::pairwise_distance_out\n")
    if (metric < 0) throw std::invalid_argument("Invalid metric");
    else if (metric == Cosine)
    {
        torch::mm_out(Out, X, Y.t());
        Out.div_((torch::norm(X, 2, 1, true).mul(torch::norm(Y, 2, 1, true).t()) + 1e-8)).mul_(-1.).add_(1.).clamp_(0., 2.);
    }
    else{
        torch::norm_out(Out, X.unsqueeze(1) - Y.unsqueeze(0), metric, 2);
    }
    return Out;
}

torch::Tensor& 
m00nny::compute_distance_matrix_out
(torch::Tensor& Out, torch::Tensor& X, torch::Tensor& Y, int64_t batchsize, int8_t metric)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: m00nny::compute_distance_matrix_out\n")
    if (batchsize <= 0) throw std::invalid_argument("compute_distance_matrix: batchsize must be positive");
    int64_t _x_size = X.size(0);
    torch::Tensor _batched_Out;
    torch::Tensor _batched_X;
    for (int64_t i = 0; i < _x_size; i += batchsize)
    {
        _batched_Out = Out.slice(0, i, i+batchsize, 1);
        _batched_X   =   X.slice(0, i, i+batchsize, 1);
        pairwise_distance_out(_batched_Out, _batched_X, Y, metric);
    }
    return Out;
}

const char*
m00nny::metric_str(int8_t metric)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: m00nny::metric_str\n")
    if (metric < 0) throw std::invalid_argument("Invalid metric");
    for (const auto& entry : kMetricNames)
        if (entry.second == metric) return entry.first;
    // Any other p names the Lp norm. (The name is returned from a per-thread buffer, not allocated.)
    thread_local char lp_name[24];
    std::snprintf(lp_name, sizeof(lp_name), "L%d norm", (int)metric);
    return lp_name;
}

bool
m00nny::metric_arg
(PyObject* o, const char* name, int8_t dflt, int8_t& out)
{
    if (pyarg::given(o) && !PyLong_Check(o) && !PyUnicode_Check(o))
    {
        PyErr_Format(PyExc_TypeError, "%s must be a metric name or a non-negative int", name);
        return false;
    }
    if (pyarg::given(o) && PyLong_Check(o))
    {
        long p = PyLong_AsLong(o);
        if (PyErr_Occurred()) return false;
        if (p < 0 || p > INT8_MAX) { PyErr_Format(PyExc_ValueError, "%s as an int is the p of an Lp norm: 0 <= p <= %d", name, INT8_MAX); return false; }
        out = (int8_t)p;
        return true;
    }
    return pyarg::to_choice(o, name, kMetricNames, dflt, out);
}

PyObject*
dist_pairwise_distance
(PyObject *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: dist_pairwise_distance\n")
    const char *kwlist[] = {"X", "Y", "metric", NULL};
    PyObject *pyX = nullptr, *pyY = nullptr, *py_metric = nullptr;
    torch::Tensor X, Y;
    int8_t metric;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|$O", (char**)kwlist, &pyX, &pyY, &py_metric)) return NULL;
    if (!m00nny::pyarg::to_tensor(pyX, "X", X) || !m00nny::pyarg::to_tensor(pyY, "Y", Y)
        || !m00nny::metric_arg(py_metric, "metric", m00nny::Euclidean, metric)) return NULL;
    LIBM00NNY_PY_TRY { return THPVariable_Wrap(m00nny::pairwise_distance(X, Y, metric)); }
    LIBM00NNY_PY_CATCH(NULL)
}

PyObject*
dist_compute_distance_matrix
(PyObject *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: dist_compute_distance_matrix\n")
    const char *kwlist[] = {"X", "Y", "batchsize", "metric", NULL};
    PyObject *pyX = nullptr, *pyY = nullptr, *py_batchsize = nullptr, *py_metric = nullptr;
    torch::Tensor X, Y;
    int64_t batchsize;
    int8_t metric;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|$OO", (char**)kwlist, &pyX, &pyY, &py_batchsize, &py_metric)) return NULL;
    if (!m00nny::pyarg::to_tensor(pyX, "X", X) || !m00nny::pyarg::to_tensor(pyY, "Y", Y)
        || !m00nny::pyarg::to_int(py_batchsize, "batchsize", 1024, batchsize)
        || !m00nny::metric_arg(py_metric, "metric", m00nny::Euclidean, metric)) return NULL;
    LIBM00NNY_PY_TRY { return THPVariable_Wrap(m00nny::compute_distance_matrix(X, Y, batchsize, metric)); }
    LIBM00NNY_PY_CATCH(NULL)
}
