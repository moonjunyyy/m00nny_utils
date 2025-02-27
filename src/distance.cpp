#include "distance.h"
    
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
    char metric_str[20] = "";
    if (metric < 0) throw std::invalid_argument("Invalid metric");
    else if (metric == Cosine)    strcpy(metric_str, "cosine");    // Cosine similarity
    else if (metric == Manhattan) strcpy(metric_str, "manhattan"); // Manhattan distance
    else if (metric == Euclidean) strcpy(metric_str, "euclidean"); // Euclidean distance
    else sprintf(metric_str, "L%d norm", metric); // Lp norm
    return strdup(metric_str);
}

PyObject*
dist_pairwise_distance
(PyObject *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: dist_pairwise_distance\n")
    const char *kwlist[] = {"X", "Y", "metric", NULL};
    PyObject *pyX, *pyY, *py_metric;
    torch::Tensor X, Y;
    int8_t metric;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|$O", (char**)kwlist, &pyX, &pyY, &py_metric)) return NULL;
    int64_t _stack = 0;
    IfSaveState(_stack, (pyX == nullptr) || Py_IsNone(pyX), [](){ PyErr_SetString(PyExc_TypeError, "X is required."); })
    IfSaveState(_stack, (pyY == nullptr) || Py_IsNone(pyY), [](){ PyErr_SetString(PyExc_TypeError, "Y is required."); })
    IfSaveState(_stack, (py_metric == nullptr) || Py_IsNone(py_metric), [&py_metric](){ py_metric = PyLong_FromLong(m00nny::Euclidean); })

    X = THPVariable_Unpack(pyX);
    Y = THPVariable_Unpack(pyY);
    metric = PyLong_AsLong(py_metric);
    ExitGILScope

    // Call the C++ pairwise_distance function.
    torch::Tensor _dist = m00nny::pairwise_distance(X, Y, metric);
    PyObject* py_dist;
    InitGILScope
    py_dist = THPVariable_Wrap(_dist);
    ExitGILScope
    return py_dist;
}

PyObject*
dist_compute_distance_matrix
(PyObject *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("  C++: dist_compute_distance_matrix\n")
    const char *kwlist[] = {"X", "Y", "batchsize", "metric", NULL};
    PyObject *pyX, *pyY, *py_batchsize, *py_metric;
    torch::Tensor X, Y;
    int64_t batchsize;
    int8_t metric;
    int64_t _stack = 0;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|$OO", (char**)kwlist, &pyX, &pyY, &py_batchsize, &py_metric)) return NULL;
    IfSaveState(_stack, (pyX == nullptr) || Py_IsNone(pyX), [](){ PyErr_SetString(PyExc_TypeError, "X is required."); })
    IfSaveState(_stack, (pyY == nullptr) || Py_IsNone(pyY), [](){ PyErr_SetString(PyExc_TypeError, "Y is required."); })
    IfSaveState(_stack, (py_batchsize == nullptr) || Py_IsNone(py_batchsize), [&py_batchsize](){ py_batchsize = PyLong_FromLong(1024); })
    IfSaveState(_stack, (py_metric == nullptr) || Py_IsNone(py_metric), [&py_metric](){ py_metric = PyLong_FromLong(m00nny::Euclidean); })

    Py_XINCREF(pyX); Py_XINCREF(pyY);
    X = THPVariable_Unpack(pyX);
    Y = THPVariable_Unpack(pyY);
    batchsize = PyLong_AsLong(py_batchsize);
    metric = PyLong_AsLong(py_metric);
    ExitGILScope

    // Call the C++ compute_distance_matrix function.
    torch::Tensor _dist = m00nny::compute_distance_matrix(X, Y, batchsize, metric);
    PyObject* py_dist;
    InitGILScope
    py_dist = THPVariable_Wrap(_dist);
    Py_XDECREF(pyX); Py_XDECREF(pyY);
    ExitGILScope
    return py_dist;
}
