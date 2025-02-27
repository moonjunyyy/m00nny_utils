#include "kmeans.h"

/*
C++ class implementation
*/

m00nny::KMeans::KMeans
(int64_t n_clusters, int64_t max_iter, int64_t batchsize, int64_t random_state, int8_t init_method, int8_t dist_method)
{
    this->n_clusters   = n_clusters;
    this->max_iter     = max_iter;
    this->batch_size   = batchsize;
    this->random_state = random_state;
    this->init_method  = init_method;
    this->dist_method  = dist_method;
}

template <typename _T>
torch::Tensor&
m00nny::KMeans::fit_predict
(_T&& X)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::KMeans::fit_predict\n")
    auto _X = std::forward<_T>(X);
    this->fit(_X);
    return this->predict(_X);
}

template <typename _T>
torch::Tensor&
m00nny::KMeans::predict
(_T&& X)
{ 
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::KMeans::predict\n")
    torch::Tensor _X = std::forward<_T>(X);
    this->update_labels(_X);
    return *(this->labels);
}

template <typename _T>
void
m00nny::KMeans::fit
(_T&& X)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::KMeans::fit\n")
    auto _X = std::forward<_T>(X);
    this->init_centers(_X);
    torch::Tensor old_centroids = torch::empty({this->n_clusters, X.size(1)}, X.options());
    for (int64_t i = 0; i < this->max_iter; i++)
    {
        old_centroids.copy_(*(this->centroids));
        this->update_labels (_X);
        this->update_centers(_X);
        if (old_centroids.equal(*(this->centroids))) break;
    }
    if (this->centroids->size(0) != this->n_clusters)
    {
        printf("Error: The number of clusters is not equal to the number of centroids.\n");
        return;
    } // If the number of clusters is not equal to the number of centroids, return by error.
}

void
m00nny::KMeans::init_centers
(torch::Tensor& X)
{
    torch::NoGradGuard no_grad;
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::KMeans::init_centers\n")
    this->centroids.reset(); this->labels.reset(); this->distances.reset(); this->mask.reset();
    this->centroids = std::make_unique<torch::Tensor>(torch::empty({this->n_clusters, X.size(1)}, X.options()));
    this->labels    = std::make_unique<torch::Tensor>(torch::empty({X.size(0)}, X.options().dtype(torch::kLong)));
    this->distances = std::make_unique<torch::Tensor>(torch::empty({X.size(0), this->n_clusters}, X.options()));
    this->mask      = std::make_unique<torch::Tensor>(torch::empty({X.size(0)}, X.options().dtype(torch::kBool)));

    if      (this->init_method == Random) this->centroids->copy_(X.index({torch::randperm(X.size(0)).slice(0, 0, this->n_clusters)}));
    else if (this->init_method == KMeansPP)
    {
        this->centroids->index({0}).copy_(X.index({torch::randint(X.size(0), {1}).item<int64_t>()})); // Choose the first centroid randomly
        torch::Tensor _distances;
        torch::Tensor _centroids;
        torch::Tensor min_dist = torch::zeros({X.size(0)}, X.options());
        torch::Tensor min_idx  = torch::empty({X.size(0)}, X.options().dtype(torch::kLong));
        for (int64_t i = 1; i < this->n_clusters; i++)
        {
            _distances = this->distances->slice(1,i-1,i,1);
            _centroids = this->centroids->slice(0,i-1,i,1);
            compute_distance_matrix_out(_distances, X, _centroids, this->batch_size, this->dist_method); // Compute the distance matrix
            torch::min_out(min_dist, min_idx, this->distances->slice(1,0,i,1), 1, false); // Get the minimum distance to the nearest centroid
            this->centroids->index({i}).copy_(X.index({(min_dist).multinomial(1).item<int64_t>()})); // Choose the next centroid
        }
    }
    else {printf("Error: Unknown init_method\n"); return;}
}

void
m00nny::KMeans::update_centers
(torch::Tensor& X)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::KMeans::update_centers\n")
    torch::NoGradGuard no_grad;
    for (int64_t i = 0; i < this->n_clusters; i++)
    {
        this->mask = std::make_unique<torch::Tensor>(this->labels->eq(i));
        if (this->mask->sum().item<int64_t>() == 0) continue;
        torch::Tensor _masked_centroids = this->centroids->index({i});
        _masked_centroids.copy_(X.index({*(this->mask)}).mean(0));
    }
}

void
m00nny::KMeans::update_labels
(torch::Tensor& X)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::KMeans::update_labels\n")
    torch::NoGradGuard no_grad;
    torch::argmin_out(*(this->labels), compute_distance_matrix(X, *(this->centroids), this->batch_size, this->dist_method), 1);
}

const char*
m00nny::KMeans::init_method_str
()
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::KMeans::init_method_str\n")
    if      (this->init_method == KMeansPP) return "kmeans++";
    else if (this->init_method == Random)   return "random";
    else return "unknown";
}

const char*
m00nny::KMeans::dist_method_str
()
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::KMeans::dist_method_str\n")
    return metric_str(this->dist_method);
}

/*
Python C API Functions
*/

PyObject*
KMeans_alloc
(PyTypeObject *type, Py_ssize_t nitems)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__alloc__\n")
    // Allocate memory for the class.
    PyObject* ret;
    InitGILScope
    ret = PyType_GenericAlloc(type, nitems);
    if (ret == NULL) return NULL;
    ExitGILScope
    // This is not a container class,
    // so we don't need to allocate memory with Py_ssiz_t nitems.
    return ret;
}

PyObject* 
KMeans_new
(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__new__\n")
    // Generate a new instance of the class.
    PyObject* self;
    InitGILScope
    self = PyType_GenericNew(type, args, kwds);
    ExitGILScope

    ((m00nny::KMeans*)self)->centroids = std::make_unique<torch::Tensor>(torch::empty({0, 0}));
    ((m00nny::KMeans*)self)->labels    = std::make_unique<torch::Tensor>(torch::empty({0}));
    ((m00nny::KMeans*)self)->distances = std::make_unique<torch::Tensor>(torch::empty({0, 0}));
    ((m00nny::KMeans*)self)->mask      = std::make_unique<torch::Tensor>(torch::empty({0}));
    // If the class always make new instances, (like non-singleton classes)
    // no special initialization is required.
    return self;
}

int
KMeans_init
(m00nny::KMeans *self, PyObject *args, PyObject *kwds)
{
    // Initialization of the instance of the class.
    // the tp_new function is called first, and allocates memory for the class.
    // Then, the tp_init function is called to initialize the instance.
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__init__\n")

    // Initialize the pointers to nullptr to get the parameters.
    PyObject *n_clusters      = nullptr,
             *max_iter        = nullptr,
             *batchsize       = nullptr,
             *init_method_str = nullptr,
             *dist_method_str = nullptr,
             *random_state    = nullptr;

    const char* kwlist[] = {"n_clusters",
                            "max_iter",
                            "batchsize",
                            "init_method",
                            "dist_method",
                            "random_state",
                            NULL}; // Sentinel
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOOOOO", (char**)kwlist,
                                    &n_clusters,
                                    &max_iter,
                                    &batchsize,
                                    &init_method_str,
                                    &dist_method_str,
                                    &random_state)) return -1;
    int64_t _stack = 0;
    IfSaveState(_stack, (n_clusters      ==nullptr) || Py_IsNone(n_clusters),      [](){ PyErr_SetString (PyExc_TypeError, "n_clusters is required.");})
    IfSaveState(_stack, (max_iter        ==nullptr) || Py_IsNone(max_iter),        [&max_iter]()       { max_iter        = PyLong_FromLong(1000); })
    IfSaveState(_stack, (batchsize       ==nullptr) || Py_IsNone(batchsize),       [&batchsize]()      { batchsize       = PyLong_FromLong(1024); })
    IfSaveState(_stack, (init_method_str ==nullptr) || Py_IsNone(init_method_str), [&init_method_str](){ init_method_str = PyUnicode_FromString("kmeans++"); })
    IfSaveState(_stack, (dist_method_str ==nullptr) || Py_IsNone(dist_method_str), [&dist_method_str](){ dist_method_str = PyUnicode_FromString("euclidean"); })
    IfSaveState(_stack, (random_state    ==nullptr) || Py_IsNone(random_state),    [&random_state]()   { random_state    = PyLong_FromLong([](){ std::random_device rd; std::mt19937 gen(rd()); return gen(); }()); })

    // We have strong references to the objects, so we don't need to Py_INCREF.
    PyObject_GenericSetAttrWithString((PyObject*)self, "n_clusters",   n_clusters);
    PyObject_GenericSetAttrWithString((PyObject*)self, "max_iter",     max_iter);
    PyObject_GenericSetAttrWithString((PyObject*)self, "batch_size",   batchsize);
    PyObject_GenericSetAttrWithString((PyObject*)self, "init_method",  init_method_str);
    PyObject_GenericSetAttrWithString((PyObject*)self, "dist_method",  dist_method_str);
    PyObject_GenericSetAttrWithString((PyObject*)self, "random_state", random_state);

    IfRestoreState(_stack, [&random_state]()   { Py_XDECREF(random_state); })
    IfRestoreState(_stack, [&dist_method_str](){ Py_XDECREF(dist_method_str); })
    IfRestoreState(_stack, [&init_method_str](){ Py_XDECREF(init_method_str); })
    IfRestoreState(_stack, [&batchsize]()      { Py_XDECREF(batchsize); })
    IfRestoreState(_stack, [&max_iter]()       { Py_XDECREF(max_iter); })
    IfRestoreState(_stack, [&n_clusters]()     { Py_XDECREF(n_clusters); })
    ExitGILScope
    return 0;
}

void
KMeans_finalize
(m00nny::KMeans *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__finalize__\n")
    // Finalize the instance of the class.
    // This function is called the end of the life cycle of the instance.
    // This not means the deallocation of the memory by garbage collector.
    // In this case, nothing to do.
    return;
}

void
KMeans_free
(m00nny::KMeans *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__free__\n")
    // Free the memory for an instance of the class.
    // This function is called when the instance is deallocated.
    // If there is no shared memory between instances,
    // this function not necessaryly to do something.
    return;
}

void
KMeans_dealloc
(m00nny::KMeans *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__dealloc__\n")
    // Deallocating the memory for the class.
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject*
KMeans_call(m00nny::KMeans *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__call__\n")
    // This function is called when the instance is called.
    // Get the parameter, assuming it is a tensor.
    // Call the fit_predict function.
    return KMeans_fit_predict(self, args, kwds);
}

PyObject*
KMeans_str(m00nny::KMeans *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__str__\n")
    // This function is called when the instance is converted to a string.
    return PyUnicode_FromFormat(
        "KMeans(n_clusters=%d, max_iter=%d, batch_size=%d, random_state=%d, init_method=%s, dist_method=%s)",
        self->n_clusters, self->max_iter, self->batch_size, self->random_state, self->init_method_str(), self->dist_method_str());
}

PyObject*
KMeans_repr(m00nny::KMeans *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__repr__\n")
    // This function is called when the instance is printed.
    return KMeans_str(self);
}

/*
C++ Class Wrapper Functions
*/

PyObject*
KMeans_fit
(m00nny::KMeans* self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.fit\n")
    // This function is called when the fit is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *pyX;
    const char* kwlist[] = {"x", NULL};

    // Get the GIL state for parsing the Python arguments.
    torch::Tensor X;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    // Call the C++ fit function.
    self->fit(X);
    InitGILScope
    Py_XDECREF(pyX);
    ExitGILScope
    return Py_None;
}

PyObject*
KMeans_fit_predict
(m00nny::KMeans* self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.fit_predict\n")
    // This function is called when the fit_predict is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *pyX;
    const char* kwlist[] = {"x", NULL};
    
    // Get the GIL state for parsing the Python arguments.
    torch::Tensor X;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    // Call the C++ fit_predict function.
    torch::Tensor ret = self->fit_predict(X);
    
    // Return the result.
    PyObject* pyret;
    InitGILScope
    pyret = THPVariable_Wrap(ret);
    Py_XDECREF(pyX);
    ExitGILScope
    return pyret;
}

PyObject*
KMeans_predict
(m00nny::KMeans* self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.predict\n")
    // This function is called when the predict is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *pyX;
    const char* kwlist[] = {"x", NULL};
    
    // Get the GIL state for parsing the Python arguments.
    torch::Tensor X;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    // Call the C++ predict function.
    torch::Tensor& ret = self->predict(X);
    
    // Return the result.
    PyObject* pyret;
    InitGILScope
    pyret = THPVariable_Wrap(ret);
    Py_XDECREF(pyX);
    ExitGILScope
    return pyret;
}

PyObject*
KMeans_init_centers
(m00nny::KMeans* self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.init_centers\n")
    // This function is called when the init_centers is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *pyX;
    const char* kwlist[] = {"x", NULL};
    
    // Get the GIL state for parsing the Python arguments.
    torch::Tensor X;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    // Call the C++ init_centers function.
    self->init_centers(X);
    InitGILScope
    Py_XDECREF(pyX);
    ExitGILScope
    return Py_None;
}

PyObject*
KMeans_update_centers
(m00nny::KMeans* self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.update_centers\n")
    // This function is called when the update_centers is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *pyX;
    const char* kwlist[] = {"x", NULL};
    
    // Get the GIL state for parsing the Python arguments.
    torch::Tensor X;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    // Call the C++ update_centers function.
    self->update_centers(X);
    InitGILScope
    Py_XDECREF(pyX);
    ExitGILScope
    return Py_None;
}

PyObject*
KMeans_update_labels
(m00nny::KMeans* self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.update_labels\n")
    // This function is called when the update_labels is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *pyX;
    const char* kwlist[] = {"x", NULL};
    
    // Get the GIL state for parsing the Python arguments.
    torch::Tensor X;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    // Call the C++ update_labels function.
    self->update_labels(X);
    InitGILScope
    Py_XDECREF(pyX);
    ExitGILScope
    return Py_None;
}

PyObject*
KMeans_compute_distance_matrix
(m00nny::KMeans* self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.compute_distance_matrix\n")
    // This function is called when the compute_distance_matrix is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *pyX;
    const char* kwlist[] = {"x", NULL};
    
    // Get the GIL state for parsing the Python arguments.
    torch::Tensor X;
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    // Call the C++ compute_distance_matrix function.
    torch::Tensor ret = m00nny::compute_distance_matrix(X, *(self->centroids), self->batch_size, self->dist_method);

    // Return the result.
    PyObject* pyret;
    InitGILScope
    pyret = THPVariable_Wrap(ret);
    Py_XDECREF(pyX);
    ExitGILScope
    return pyret;
}

/*
    C++ Class Getters and Setters
*/

PyObject*
KMeans_get_init_method
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__get__(init_method)\n")
    PyObject* pyret;
    InitGILScope
    pyret = PyUnicode_FromString(((m00nny::KMeans*)self)->init_method_str());
    ExitGILScope
    return pyret;
}

PyObject*
KMeans_get_dist_method
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__get__(dist_method)\n")
    PyObject* pyret;
    InitGILScope
    pyret = PyUnicode_FromString(((m00nny::KMeans*)self)->dist_method_str());
    ExitGILScope
    return pyret;
}

int
KMeans_set_init_method
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__set__(init_method)\n")
    InitGILScope
    if (PyUnicode_Check(value))
    {
        const char* value_str = PyUnicode_AsUTF8(value);
        if      (!strcmp(value_str, "kmeans++")) ((m00nny::KMeans*)self)->init_method = m00nny::KMeans::KMeansPP;
        else if (!strcmp(value_str, "random"))   ((m00nny::KMeans*)self)->init_method = m00nny::KMeans::Random;
        else 
        {
            char msg[200] = "Unknown init_method. The init_method must be one of 'kmeans++' or 'random': ";
            strncat(msg, PyUnicode_AsUTF8(value), 200 - 77); // 200 - 72 is the left space for the message buffer.
            PyErr_SetString(PyExc_ValueError, (const char*)msg);
            return -1;
        }
    }
    else if (PyLong_Check(value))
    {
        int8_t value_int = PyLong_AsLong(value);
        if (value_int < 0 || value_int > 1)
        {
            char msg[200] = "Unknown init_method. The init_method should be 0: 'kmeans++' or 1: 'random': ";
            strncat(msg, PyUnicode_AsUTF8(value), 200 - 72); // 200 - 77 is the left space for the message buffer.
            PyErr_SetString(PyExc_ValueError, (const char*)msg);
            return -1;
        }
        ((m00nny::KMeans*)self)->init_method = value_int;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "The init_method must be a string or an integer.");
        return -1;
    }
    ExitGILScope
    return 0;
}

int
KMeans_set_dist_method
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__set__(dist_method)\n")
    InitGILScope
    if (PyUnicode_Check(value))
    {
        const char* value_str = PyUnicode_AsUTF8(value);
        if      (!strcmp(value_str, "euclidean")) ((m00nny::KMeans*)self)->dist_method = m00nny::Euclidean;
        else if (!strcmp(value_str, "cosine"))    ((m00nny::KMeans*)self)->dist_method = m00nny::Cosine;
        else if (!strcmp(value_str, "manhattan")) ((m00nny::KMeans*)self)->dist_method = m00nny::Manhattan;
        else 
        {
            char msg[200] = "Unknown init_method. The dist_method must be one of 'euclidean', 'cosine', or 'manhattan': ";
            strncat(msg, PyUnicode_AsUTF8(value), 200 - 92); // 200 - 92 is the left space for the message buffer.
            PyErr_SetString(PyExc_ValueError, (const char*)msg);
            return -1;
        }
    }
    else if (PyLong_Check(value))
    {
        int8_t value_int = PyLong_AsLong(value);
        if (value_int < 0)
        {
            char msg[200] = "Unknown init_method. The dist_method should be a non-negative integer: ";
            strncat(msg, PyUnicode_AsUTF8(value), 200 - 72); // 200 - 77 is the left space for the message buffer.
            PyErr_SetString(PyExc_ValueError, (const char*)msg);
            return -1;
        }
        ((m00nny::KMeans*)self)->dist_method = value_int;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "The dist_method must be a string or an integer.");
        return -1;
    }
    ExitGILScope
    return 0;
}

PyObject*
KMeans_get_centroids
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__get__(centroids)\n")
    PyObject* pyret;
    InitGILScope
    pyret = THPVariable_Wrap(*(((m00nny::KMeans*)self)->centroids));
    ExitGILScope
    return pyret;
}

PyObject*
KMeans_get_labels
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__get__(labels)\n")
    PyObject* pyret;
    InitGILScope
    pyret = THPVariable_Wrap(*(((m00nny::KMeans*)self)->labels));
    ExitGILScope
    return pyret;
}

PyObject*
KMeans_get_distances
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__get__(distances)\n")
    PyObject* pyret;
    InitGILScope
    pyret = THPVariable_Wrap(*(((m00nny::KMeans*)self)->distances));
    ExitGILScope
    return pyret;
}

PyObject*
KMeans_get_mask
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__get__(mask)\n")
    PyObject* pyret;
    InitGILScope
    pyret = THPVariable_Wrap(*(((m00nny::KMeans*)self)->mask));
    ExitGILScope
    return pyret;
}

int
KMeans_set_centroids
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__set__(centroids)\n")
    InitGILScope
    *(((m00nny::KMeans*)self)->centroids) = THPVariable_Unpack(value);
    ExitGILScope
    return 0;
}

int
KMeans_set_labels
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__set__(labels)\n")
    InitGILScope
    *(((m00nny::KMeans*)self)->labels)= THPVariable_Unpack(value);
    ExitGILScope
    return 0;
}

int
KMeans_set_distances
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__set__(distances)\n")
    InitGILScope
    *(((m00nny::KMeans*)self)->distances) = THPVariable_Unpack(value);
    ExitGILScope
    return 0;
}

int
KMeans_set_mask
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__set__(mask)\n")
    InitGILScope
    *(((m00nny::KMeans*)self)->mask) = THPVariable_Unpack(value);
    ExitGILScope
    return 0;
}