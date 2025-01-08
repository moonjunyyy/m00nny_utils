#include "kmeans.h"

/*
C++ class implementation
*/

KMeans::KMeans(int n_clusters, int max_iter, int batchsize, int random_state, int init_method, int dist_method, double epsilon)
{
    this->n_clusters   = n_clusters;
    this->max_iter     = max_iter;
    this->batch_size   = batchsize;
    this->random_state = random_state;
    this->init_method  = init_method;
    this->dist_method  = dist_method;
    this->epsilon      = epsilon;
}

template <typename _T>
at::Tensor&
KMeans::fit_predict(_T&& X)
{ 
    auto _X = std::forward<_T>(X);
    this->fit(_X);
    return this->predict(_X);
}

template <typename _T>
at::Tensor&
KMeans::predict(_T&& X)
{ 
    at::Tensor _X = std::forward<_T>(X);
    this->update_labels(_X);
    return this->labels;
}

template <typename _T>
void
KMeans::fit(_T&& X)
{
    torch::NoGradGuard no_grad;
    auto _X = std::forward<_T>(X);
    this->init_centers(_X);
    at::Tensor old_centroids = torch::empty({this->n_clusters, X.size(1)}, X.options());
    for (int i = 0; i < this->max_iter; i++)
    {
        old_centroids.copy_(this->centroids);
        this->update_labels (_X);
        this->update_centers(_X);
        if (old_centroids.equal(this->centroids)) break;
    }
    if (this->centroids.size(0) != this->n_clusters)
    {
        printf("Error: The number of clusters is not equal to the number of centroids.\n");
        return;
    } // If the number of clusters is not equal to the number of centroids, return by error.
}

void
KMeans::init_centers(at::Tensor& X)
{
    this->centroids = at::empty({this->n_clusters, X.size(1)}, X.options());
    this->labels    = at::empty({X.size(0)}, X.options().dtype(at::kInt));
    this->distances = at::empty({X.size(0), this->n_clusters}, X.options());
    // this->mask      = at::empty({X.size(0)}, X.options().dtype(at::kBool));

    if      (this->init_method == Random) this->centroids.copy_(X.index({torch::randperm(X.size(0)).slice(0, 0, this->n_clusters)}));
    else if (this->init_method == KMeansPP)
    {
        this->centroids.index({0}).copy_(X.index({torch::randperm(X.size(0)).slice(0, 0, 1)}));                  // Choose the first centroid randomly
        at::Tensor _distances = this->distances.slice(1,0,1,1);
        at::Tensor _centroids = this->centroids.slice(0,0,1,1);
        at::Tensor min_dist = torch::zeros({X.size(0)}, X.options()) + this->epsilon;
        at::Tensor min_idx  = torch::empty({X.size(0)}, X.options().dtype(torch::kInt64));
        compute_distance_matrix_out(_distances, X, _centroids);                                                  // Compute the distance matrix
        for (int i = 0; i < (this->n_clusters - 1); i++)
        {   
            _distances = this->distances.slice(1,0,i+1,1);
            _centroids = this->centroids.slice(0,0,i+1,1);
            compute_distance_matrix_out(_distances, X, _centroids);                                              // Compute the distance matrix
            torch::min_out(min_dist, min_idx, this->distances.slice(1,0,i+1,1), 1, false);                       // Get the minimum distance to the nearest centroid
            this->centroids.index({i+1}).copy_(X.index({(min_dist/min_dist.sum()).multinomial(1).item<int>()})); // Choose the next centroid
        }
    }
    else {printf("Error: Unknown init_method\n"); return;}
}

void
KMeans::update_centers(at::Tensor& X)
{
    torch::NoGradGuard no_grad;
    for (int i = 0; i < this->n_clusters; i++)
    {
        this->mask = this->labels == i;
        if (this->mask.sum().item<int>() == 0) continue;
        at::Tensor _masked_centroids = this->centroids.index({i});
        _masked_centroids.copy_(X.index({this->mask}).mean(0));
    }
}

void
KMeans::update_labels(at::Tensor& X)
{
    torch::NoGradGuard no_grad;
    torch::argmin_out(this->labels, this->compute_distance_matrix(X, this->centroids), 1);
}

at::Tensor&
KMeans::compute_distance_matrix(at::Tensor& X)
{
    torch::NoGradGuard no_grad;
    at::Tensor _batched_distance;
    at::Tensor _batched_X;
    for (int64_t i = 0; i < X.size(0); i += this->batch_size) 
    {
        _batched_distance = this->distances.slice(0, i, i+this->batch_size, 1);
        _batched_X        = X.slice(0, i, i+this->batch_size, 1);
        this->pairwise_distance_out( _batched_distance, _batched_X, this->centroids);
    }
    return this->distances;
}

at::Tensor&
KMeans::compute_distance_matrix(at::Tensor& X, at::Tensor& Y)
{
    torch::NoGradGuard no_grad;
    at::Tensor _batched_distance;
    at::Tensor _batched_X;
    for (int64_t i = 0; i < X.size(0); i += this->batch_size) 
    {
        _batched_distance = this->distances.slice(0, i, i+this->batch_size, 1);
        _batched_X        = X.slice(0, i, i+this->batch_size, 1);
        this->pairwise_distance_out(_batched_distance, _batched_X, Y);
    }
    return this->distances;
}

at::Tensor&
KMeans::compute_distance_matrix_out(at::Tensor& Out, at::Tensor& X, at::Tensor& Y)
{
    torch::NoGradGuard no_grad;
    int64_t _x_size = X.size(0);
    at::Tensor _batched_Out;
    at::Tensor _batched_X;
    for (int64_t i = 0; i < _x_size; i += this->batch_size)
    {
        _batched_Out = Out.slice(0, i, i+this->batch_size, 1);
        _batched_X   =   X.slice(0, i, i+this->batch_size, 1);
        this->pairwise_distance_out(_batched_Out, _batched_X, Y);
    }
    return Out;
}

at::Tensor
KMeans::pairwise_distance(at::Tensor& X, at::Tensor& Y)
{   
    torch::NoGradGuard no_grad;
    if (this->dist_method == Euclidean)
    {
        at::Tensor _dist = torch::norm(X.unsqueeze(1) - Y.unsqueeze(0), 2, 2);
        return _dist;
    }
    else if (this->dist_method == Cosine)
    {
        at::Tensor _dist = 1 - torch::clamp(torch::mm(X, Y.t()) / (torch::norm(X, 2, 1, true) * torch::norm(Y, 2, 1, true).t() + this->epsilon), -1.0, 1.0);
        return _dist;
    }
    else if (this->dist_method == Manhattan)
    {
        at::Tensor _dist = torch::norm(X.unsqueeze(1) - Y.unsqueeze(0), 1, 2);
        return _dist;
    }
    else
    {
        printf("Error: Unknown distance method\n");
        return torch::empty({0, 0});
    }
}

at::Tensor&
KMeans::pairwise_distance_out(at::Tensor& Out, at::Tensor& X, at::Tensor& Y){
    if (this->dist_method == Euclidean) 
        torch::norm_out(Out, X.unsqueeze(1) - Y.unsqueeze(0), 2, 2);
    else if (this->dist_method == Cosine)
    {
        torch::mm_out(Out, X, Y.t());
        Out.div_(torch::norm(X, 2, 1, true).mul(torch::norm(Y, 2, 1, true).t()) + this->epsilon).sub_(1.0).mul_(-1.0);
    }
    else if (this->dist_method == Manhattan)
        torch::norm_out(Out, X.unsqueeze(1) - Y.unsqueeze(0), 1, 2);
    else
    {
        printf("Error: Unknown distance method\n");
        return Out;
    }
    return Out;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

PyObject*
KMeans_alloc(PyTypeObject *type, Py_ssize_t nitems)
{
    // Allocate memory for the class.
    KMeans* ret = PyObject_New(KMeans, type);
    if (ret == NULL) return NULL;
    // This is not a container class,
    // so we don't need to allocate memory with Py_ssiz_t nitems.
    ret->x_attr = PyDict_New();
    if (ret->x_attr == NULL) return NULL;
    return (PyObject*)ret;
}

PyObject* 
KMeans_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    // Generate a new instance of the class.
    KMeans* self = (KMeans*)KMeans_alloc(type, 0);
    // If the class always make new instances, (like non-singleton classes)
    // no special initialization is required.
    if (self != NULL) KMeans_init(self, args, kwds);
    return (PyObject*)self;
}

int
KMeans_init(KMeans *self, PyObject *args, PyObject *kwds)
{
    // Initialization of the instance of the class.
    // the tp_new function is called first, and allocates memory for the class.
    // Then, the tp_init function is called to initialize the instance.

    // Initialize the pointers to nullptr to get the parameters.
    PyObject *n_clusters=nullptr, *max_iter=nullptr, *batchsize=nullptr, *random_state=nullptr;
    char *init_method_str=nullptr, *dist_method_str=nullptr;
    float epsilon = 1e-8;
    const char* kwlist[] = {"n_clusters",
                            "max_iter",
                            "batchsize",
                            "mode",
                            "init",
                            "seed",
                            "epsilon",
                            NULL}; // Sentinel
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOssOf", (char**)kwlist,
                                    &n_clusters,
                                    &max_iter,
                                    &batchsize,
                                    &dist_method_str,
                                    &init_method_str,
                                    &random_state,
                                    &epsilon)) return -1;

    // Get the integer values from the Python objects.
    if (Py_IsNone(n_clusters)) PyErr_SetString(PyExc_ValueError, "n_clusters is required");
    if (Py_IsNone(max_iter))   PyErr_SetString(PyExc_ValueError, "max_iter is required");
    if (Py_IsNone(batchsize))  PyErr_SetString(PyExc_ValueError, "batchsize is required");
    KMeans_setattro(self, "n_clusters", n_clusters);
    KMeans_setattro(self, "max_iter",   max_iter);
    KMeans_setattro(self, "batch_size", batchsize);

    // Get the random state, if it is not provided, generate a random number.
    long _random_state = Py_IsNone(random_state) ? [](){std::srand(std::time(0)); return (long)std::rand();}() : PyLong_AsLong(random_state);
    KMeans_setattro(self, "random_state", PyLong_FromLong(_random_state));
    // Get the initialization method and distance method from the strings.
    KMeans_setattro(self, "init_method", PyUnicode_FromString(init_method_str));
    // Get the distance method from the string.
    KMeans_setattro(self, "dist_method", PyUnicode_FromString(dist_method_str));
    return 0;
}

void
KMeans_finalize(KMeans *self)
{
    // Finalize the instance of the class.
    // This function is called the end of the life cycle of the instance.
    // This not means the deallocation of the memory by garbage collector.

    // In this case, nothing to do.
    return;
}

void
KMeans_free(KMeans *self)
{
    // Free the memory for an instance of the class.
    // This function is called when the instance is deallocated.
    // If there is no shared memory between instances,
    // this function not necessaryly to do something.
    KMeans_dealloc(self);
}

void
KMeans_dealloc(KMeans *self)
{
    // Deallocating the memory for the class.
    Py_XDECREF(self->x_attr);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

int
KMeans_setattro(KMeans *self, const char *name, PyObject *v)
{
    // Set an attribute of the instance.
    // This function is called when the instance is assigned a value.
    if (self->x_attr == NULL)
    {
        // Create a new dictionary for the attributes.
        // This is not necessary, but for the safety.
        self->x_attr = PyDict_New();
        if (self->x_attr == NULL) return -1;
    }

    // If the value is None, delete the attribute.
    if (v == NULL)
    {
        int rv = PyDict_DelItemString(self->x_attr, name);
        // If the attribute is not found, raise an AttributeError.
        if (rv < 0 && PyErr_ExceptionMatches(PyExc_KeyError))
            PyErr_SetString(PyExc_AttributeError,
                "delete non-existing Kmeans attribute");
        return rv;
    }

    // Otherwise, set the attribute.
    // This will automatically increment the reference count.
    int rv = PyDict_SetItemString(self->x_attr, name, v);
    // If the attribute name is about the main functions, set the c++ instance variables.
    if      (!strcmp(name, "n_clusters"))   self->n_clusters   = PyLong_AsLong(v);
    else if (!strcmp(name, "max_iter"))     self->max_iter     = PyLong_AsLong(v);
    else if (!strcmp(name, "batch_size"))   self->batch_size   = PyLong_AsLong(v);
    else if (!strcmp(name, "random_state")) self->random_state = PyLong_AsLong(v);
    else if (!strcmp(name, "epsilon"))      self->epsilon      = PyFloat_AsDouble(v);
    else if (!strcmp(name, "init_method"))
    {
        const char* init_method_str = PyUnicode_AsUTF8(v);
        if      (!strcmp(init_method_str, "kmeans++")) self->init_method = KMeansPP;
        else if (!strcmp(init_method_str, "random"))   self->init_method = Random;
        else 
        {
            char msg[50] = "Unknown init_method: ";
            strncat(msg, init_method_str, 28); // 28 is the left space for the message buffer.
            PyErr_SetString(PyExc_ValueError, (const char*)msg);
            return -1;
        }
    }
    else if (!strcmp(name, "dist_method"))
    {
        const char* dist_method_str = PyUnicode_AsUTF8(v);
        if      (!strcmp(dist_method_str, "euclidean")) { self->dist_method = Euclidean; }
        else if (!strcmp(dist_method_str, "cosine"))    { self->dist_method = Cosine; }
        else if (!strcmp(dist_method_str, "manhattan")) { self->dist_method = Manhattan; }
        else 
        {
            char msg[50] = "Unknown init_method: ";
            strncat(msg, dist_method_str, 28); // 28 is the left space for the message buffer.
            PyErr_SetString(PyExc_ValueError, (const char*)msg);
            return -1;
        }
    }
    return rv;
}

PyObject*
KMeans_getattro(KMeans *self, PyObject *name)
{
    // Get an attribute of the instance.
    // This function is called when the instance is accessed.
    // If the attribute is not found, raise an AttributeError.
    if (self->x_attr == NULL)
    {
        PyErr_SetString(PyExc_AttributeError,
            "Kmeans has no attribute");
        return NULL;
    }
    return PyDict_GetItem(self->x_attr, name);
}

PyObject*
KMeans_call(KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the instance is called.
    // Get the parameter, assuming it is a tensor.
    // Call the fit_predict function.
    return fit_predict(self, args, kwds);
}

PyObject*
KMeans_str(KMeans *self)
{
    // This function is called when the instance is converted to a string.
    return PyUnicode_FromFormat(
        "KMeans(n_clusters=%d, max_iter=%d, batch_size=%d, random_state=%d, init_method=%s, dist_method=%s, epsilon=%f)",
        self->n_clusters, self->max_iter, self->batch_size, self->random_state, self->init_method, self->dist_method, self->epsilon);
}

PyObject*
KMeans_repr(KMeans *self)
{
    // This function is called when the instance is printed.
    return KMeans_str(self);
}

/*
C++ Class Wrapper Functions
*/
PyObject*
fit(KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the fit method is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *X;
    const char* kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &X)) return NULL;
    // Call the fit function.
    self->fit(THPVariable_Unpack(X));
    return Py_None;
}

PyObject*
fit_predict(KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the fit_predict method is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *X;
    const char* kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &X)) return NULL;
    // Call the fit_predict function.
    return THPVariable_Wrap(self->fit_predict(THPVariable_Unpack(X)));
}

PyObject*
predict(KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the predict method is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *X;
    const char* kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &X)) return NULL;
    // Call the predict function.
    return THPVariable_Wrap(self->predict(THPVariable_Unpack(X)));
}

PyObject* init_centers(KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the init_centers method is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *X;
    const char* kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &X)) return NULL;
    // Call the init_centers function.
    at::Tensor _X = THPVariable_Unpack(X);
    self->init_centers(_X);
    return Py_None;
}

PyObject* update_centers(KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the update_centers method is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *X;
    const char* kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &X)) return NULL;
    // Call the update_centers function.
    at::Tensor _X = THPVariable_Unpack(X);
    self->update_centers(_X);
    return Py_None;
}

PyObject* update_labels(KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the update_labels method is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *X;
    const char* kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &X)) return NULL;
    // Call the update_labels function.
    at::Tensor _X = THPVariable_Unpack(X);
    self->update_labels(_X);
    return Py_None;
}

PyObject* compute_distance_matrix (KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the compute_distance_matrix method is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *X;
    const char* kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &X)) return NULL;
    // Call the compute_distance_matrix function.
    at::Tensor _X = THPVariable_Unpack(X);
    return THPVariable_Wrap(self->compute_distance_matrix(_X));
}

PyObject* get_centroids(KMeans *self, PyObject *args, PyObject *kwds)
{
    // This function is called when the get_centroids method is called.
    // Get the parameter, assuming it is a tensor.
    PyObject *X;
    const char* kwlist[] = {"x", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &X)) return NULL;
    // Return the centroids.
    return THPVariable_Wrap(self->centroids);
}

#ifdef __cplusplus
}
#endif // __cplusplus