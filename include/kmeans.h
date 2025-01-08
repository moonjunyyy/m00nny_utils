#ifndef KMEANS_H
#define KMEANS_H
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <ctime>
#include <cstdlib>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/python_variable.h>

enum InitMethod { KMeansPP, Random };
enum DistMethod { Euclidean, Cosine, Manhattan };

class KMeans {
public:    
    PyObject_HEAD;
    PyObject* x_attr; /* Attributes dictionary */

    int n_clusters;
    int max_iter;
    int batch_size;
    int random_state;
    int init_method;
    int dist_method;
    double epsilon = 1e-8;

    at::Tensor centroids;
    at::Tensor labels;
    at::Tensor distances;
    at::Tensor mask;

    KMeans() = default;
    KMeans(int n_clusters, int max_iter, int batchsize, int random_state, int init_method, int dist_method, double epsilon);
    
    template <typename _T>
    at::Tensor& fit_predict(_T&& X);

    template <typename _T>
    at::Tensor& predict(_T&& X);

    template <typename _T>
    void fit(_T&& X);

    inline void init_centers  (at::Tensor& X);
    inline void update_centers(at::Tensor& X);
    inline void update_labels (at::Tensor& X);

    inline at::Tensor  pairwise_distance      (at::Tensor& X, at::Tensor& Y);
    inline at::Tensor& compute_distance_matrix(at::Tensor& X);
    inline at::Tensor& compute_distance_matrix(at::Tensor& X, at::Tensor& Y);

    inline at::Tensor& pairwise_distance_out      (at::Tensor& Out, at::Tensor& X, at::Tensor& Y);
    inline at::Tensor& compute_distance_matrix_out(at::Tensor& Out, at::Tensor& X, at::Tensor& Y);
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
Python C API Functions
*/
static PyObject* KMeans_alloc            (PyTypeObject *type, Py_ssize_t nitems=0);            // Allocate memory for the class (C++ operations)
static PyObject* KMeans_new              (PyTypeObject *type, PyObject *args, PyObject *kwds); // Create a new object (Python operations)
static int       KMeans_init             (KMeans *self, PyObject *args, PyObject *kwds);       // Initialize the Instance (Python operations)
static void      KMeans_finalize         (KMeans *self);                                       // Finalize the Instance (Python operations)
static void      KMeans_free             (KMeans *self);                                       // Free the memory for the object (Python operations)
static void      KMeans_dealloc          (KMeans *self);                                       // Deallocate memory for the class (C++ operations)
static int       KMeans_setattro         (KMeans *self, const char *name, PyObject *v);        // Set an attribute
static PyObject* KMeans_getattro         (KMeans *self, PyObject *name);                       // Get an attribute
static PyObject* KMeans_call             (KMeans *self, PyObject *args, PyObject *kwds);       // Function when the object is called
static PyObject* KMeans_str              (KMeans *self);                                       // Function when the object is converted to string
static PyObject* KMeans_repr             (KMeans *self);                                       // Function when the object is printed

/*
C++ Class Wrapper Functions
*/
static PyObject* fit                     (KMeans *self, PyObject *args, PyObject *kwds);
static PyObject* fit_predict             (KMeans *self, PyObject *args, PyObject *kwds);
static PyObject* predict                 (KMeans *self, PyObject *args, PyObject *kwds);
static PyObject* init_centers            (KMeans *self, PyObject *args, PyObject *kwds);
static PyObject* update_centers          (KMeans *self, PyObject *args, PyObject *kwds);
static PyObject* update_labels           (KMeans *self, PyObject *args, PyObject *kwds);
static PyObject* compute_distance_matrix (KMeans *self, PyObject *args, PyObject *kwds);
static PyObject* get_centroids           (KMeans *self, PyObject *args, PyObject *kwds);

static PyMethodDef KMeans_methods[] =
{
    {"fit",                     (PyCFunction)fit,                     METH_VARARGS, "Fits the model to the data."},
    {"fit_predict",             (PyCFunction)fit_predict,             METH_VARARGS, "Fits the model to the data and then predicts the clusters."},
    {"predict",                 (PyCFunction)predict,                 METH_VARARGS, "Predicts the clusters for the data."},
    {"init_centers",            (PyCFunction)init_centers,            METH_VARARGS, "Initializes the cluster centers."},
    {"update_centers",          (PyCFunction)update_centers,          METH_VARARGS, "Updates the cluster centers."},
    {"update_labels",           (PyCFunction)update_labels,           METH_VARARGS, "Updates the labels for the data points."},
    {"compute_distance_matrix", (PyCFunction)compute_distance_matrix, METH_VARARGS, "Computes the distance matrix."},
    {"get_centroids",           (PyCFunction)get_centroids,           METH_VARARGS, "Returns the cluster centers."},
    {NULL, NULL} /* sentinel */
};

static PyMemberDef KMeans_members[] = {
    {"n_clusters",   Py_T_INT, offsetof(KMeans, n_clusters),   0, "Number of clusters"},
    {"max_iter",     Py_T_INT, offsetof(KMeans, max_iter),     0, "Maximum number of iterations"},
    {"batch_size",   Py_T_INT, offsetof(KMeans, batch_size),   0, "Batch size"},
    {"random_state", Py_T_INT, offsetof(KMeans, random_state), 0, "Random state"},
    {"init_method",  Py_T_INT, offsetof(KMeans, init_method),  0, "Initialization method"},
    {"dist_method",  Py_T_INT, offsetof(KMeans, dist_method),  0, "Distance method"},
    {NULL}  /* Sentinel */
};

PyTypeObject KMeansType =
{
    /* The ob_type field must be initialized in the module init function
     * to be portable to Windows without using C++. */
    PyVarObject_HEAD_INIT(NULL, 0)
    "KMeans",                        /*tp_name*/
    sizeof(KMeans),                  /*tp_basicsize*/
    0,                               /*tp_itemsize*/
    /* methods */
    (destructor)    KMeans_dealloc,  /*tp_dealloc*/
    0,                               /*tp_vectorcall_offset*/
    0,                               /*tp_getattr (deprecated)*/
    0,                               /*tp_setattr (deprecated)*/
    0,                               /*tp_as_async*/
    (reprfunc)      KMeans_repr,     /*tp_repr*/
    0,                               /*tp_as_number*/
    0,                               /*tp_as_sequence*/
    0,                               /*tp_as_mapping*/
    0,                               /*tp_hash*/
    (ternaryfunc)   KMeans_call,     /*tp_call*/
    (reprfunc)      KMeans_str,      /*tp_str*/
    (getattrofunc)  KMeans_getattro, /*tp_getattro*/
    (setattrofunc)  KMeans_setattro, /*tp_setattro*/
    0,                               /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,              /*tp_flags*/
    0,                               /*tp_doc*/
    0,                               /*tp_traverse*/
    0,                               /*tp_clear*/
    0,                               /*tp_richcompare*/
    0,                               /*tp_weaklistoffset*/
    0,                               /*tp_iter*/
    0,                               /*tp_iternext*/
    KMeans_methods,                  /*tp_methods*/
    KMeans_members,                  /*tp_members*/
    0,                               /*tp_getset*/
    0,                               /*tp_base*/
    0,                               /*tp_dict*/
    0,                               /*tp_descr_get*/
    0,                               /*tp_descr_set*/
    0,                               /*tp_dictoffset*/
    (initproc)      KMeans_init,     /*tp_init*/
    (allocfunc)     KMeans_alloc,    /*tp_alloc*/
    (newfunc)       KMeans_new,      /*tp_new*/
    (freefunc)      KMeans_free,     /*tp_free*/
    0,                               /*tp_is_gc*/
    0,                               /*tp_bases*/
    0,                               /*tp_mro (method resolution order) */
    0,                               /*tp_cache (no longer used) */
    0,                               /*tp_subclasses (for static builtin types this is an index)*/
    0,                               /*tp_weaklist (not used for static builtin types)*/
    0,                               /*tp_del (deprecated)*/
    /* Type attribute cache version tag. Added in version 2.6 */
    0,                               /*tp_version_tag*/
    (destructor)    KMeans_finalize, /*tp_finalize*/
    0,                               /*tp_vectorcall*/
    /* bitset of which type-watchers care about this type */
    0,                               /*tp_watched*/
};

/* --------------------------------------------------------------------- */
#define KMeansObject_Check(v) Py_IS_TYPE(v, &KMeansType)

const char *module_doc = "This module provides an KMeans interface for Pytorch";
struct PyModuleDef KMeansModule = {
    PyModuleDef_HEAD_INIT,
    "kmeans",
    module_doc,
    0,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_kmeans(void)
{
    PyObject *m;
    m = PyModule_Create(&KMeansModule);
    if (m == NULL) return nullptr;

    Py_INCREF(&KMeansType);
    PyModule_AddObject(m, "KMeans", (PyObject *)&KMeansType);
    return m;
}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // KMEANS_H