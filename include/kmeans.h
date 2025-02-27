#ifndef LIBM00NNY_KMEANS_H
#define LIBM00NNY_KMEANS_H

#include "common.h"
#include "macros.h"
#include "distance.h"

namespace m00nny {

class KMeans {
public:    
    PyObject_HEAD;

    enum KMeansInitMethod { KMeansPP, Random };

    int64_t n_clusters;
    int64_t max_iter;
    int64_t batch_size;
    int64_t random_state;
    int8_t  init_method;
    int8_t  dist_method;

    std::unique_ptr<torch::Tensor> centroids;
    std::unique_ptr<torch::Tensor> labels;
    std::unique_ptr<torch::Tensor> distances;
    std::unique_ptr<torch::Tensor> mask;

    KMeans(int64_t n_clusters, int64_t max_iter, int64_t batchsize, int64_t random_state, int8_t init_method, int8_t dist_method);
    
    template <typename _T>
    torch::Tensor& fit_predict(_T&& X);

    template <typename _T>
    torch::Tensor& predict(_T&& X);

    template <typename _T>
    void fit(_T&& X);

    void init_centers  (torch::Tensor& X);
    void update_centers(torch::Tensor& X);
    void update_labels (torch::Tensor& X);

    const char* init_method_str();
    const char* dist_method_str();
};
} // namespace m00nny

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#define PY_SSIZE_T_CLEAN
/*
Python C API Functions
*/
PyObject* KMeans_alloc            (PyTypeObject *type, Py_ssize_t nitems=0);            // Allocate memory for the class (C++ operations)
PyObject* KMeans_new              (PyTypeObject *type, PyObject *args, PyObject *kwds); // Create a new object (Python operations)
int       KMeans_init             (m00nny::KMeans *self, PyObject *args, PyObject *kwds);       // Initialize the Instance (Python operations)
void      KMeans_finalize         (m00nny::KMeans *self);                                       // Finalize the Instance (Python operations)
void      KMeans_free             (m00nny::KMeans *self);                                       // Free the memory for the object (Python operations)
void      KMeans_dealloc          (m00nny::KMeans *self);                                       // Deallocate memory for the class (C++ operations)
PyObject* KMeans_call             (m00nny::KMeans *self, PyObject *args, PyObject *kwds);       // Function when the object is called
PyObject* KMeans_str              (m00nny::KMeans *self);                                       // Function when the object is converted to string
PyObject* KMeans_repr             (m00nny::KMeans *self);                                       // Function when the object is printed

/*
C++ Class Wrapper Functions
*/
PyObject* KMeans_fit_predict             (m00nny::KMeans *self, PyObject *args, PyObject *kwds);
PyObject* KMeans_predict                 (m00nny::KMeans *self, PyObject *args, PyObject *kwds);
PyObject* KMeans_fit                     (m00nny::KMeans *self, PyObject *args, PyObject *kwds);
PyObject* KMeans_init_centers            (m00nny::KMeans *self, PyObject *args, PyObject *kwds);
PyObject* KMeans_update_centers          (m00nny::KMeans *self, PyObject *args, PyObject *kwds);
PyObject* KMeans_update_labels           (m00nny::KMeans *self, PyObject *args, PyObject *kwds);
PyObject* KMeans_compute_distance_matrix (m00nny::KMeans *self, PyObject *args, PyObject *kwds);

/*
C++ Class Getters
*/
PyObject* KMeans_get_init_method (PyObject *self, void* closure);
PyObject* KMeans_get_dist_method (PyObject *self, void* closure);
PyObject* KMeans_get_centroids   (PyObject *self, void* closure);
PyObject* KMeans_get_labels      (PyObject *self, void* closure);
PyObject* KMeans_get_distances   (PyObject *self, void* closure);
PyObject* KMeans_get_mask        (PyObject *self, void* closure);

/*
C++ Class Setters
*/
int KMeans_set_init_method (PyObject *self, PyObject *value, void* closure);
int KMeans_set_dist_method (PyObject *self, PyObject *value, void* closure);
int KMeans_set_centroids   (PyObject *self, PyObject *value, void* closure);
int KMeans_set_labels      (PyObject *self, PyObject *value, void* closure);
int KMeans_set_distances   (PyObject *self, PyObject *value, void* closure);
int KMeans_set_mask        (PyObject *self, PyObject *value, void* closure);

#ifdef __cplusplus
}
#endif // __cplusplus

static PyMethodDef KMeans_methods[] =
{
    {"fit",                     (PyCFunction)KMeans_fit,                     METH_VARARGS | METH_KEYWORDS, "Fits the model to the data."},
    {"fit_predict",             (PyCFunction)KMeans_fit_predict,             METH_VARARGS | METH_KEYWORDS, "Fits the model to the data and then predicts the clusters."},
    {"predict",                 (PyCFunction)KMeans_predict,                 METH_VARARGS | METH_KEYWORDS, "Predicts the clusters for the data."},
    {"init_centers",            (PyCFunction)KMeans_init_centers,            METH_VARARGS | METH_KEYWORDS, "Initializes the cluster centers."},
    {"update_centers",          (PyCFunction)KMeans_update_centers,          METH_VARARGS | METH_KEYWORDS, "Updates the cluster centers."},
    {"update_labels",           (PyCFunction)KMeans_update_labels,           METH_VARARGS | METH_KEYWORDS, "Updates the labels for the data points."},
    {"compute_distance_matrix", (PyCFunction)KMeans_compute_distance_matrix, METH_VARARGS | METH_KEYWORDS, "Computes the distance matrix."},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static PyMemberDef KMeans_members[] = {
    {"n_clusters",   Py_T_INT, offsetof(m00nny::KMeans, n_clusters),   0, "Number of clusters"},
    {"max_iter",     Py_T_INT, offsetof(m00nny::KMeans, max_iter),     0, "Maximum number of iterations"},
    {"batch_size",   Py_T_INT, offsetof(m00nny::KMeans, batch_size),   0, "Batch size"},
    {"random_state", Py_T_INT, offsetof(m00nny::KMeans, random_state), 0, "Random state"},
    {NULL}  /* Sentinel */
};

static PyGetSetDef KMeans_getsetters[] = {
    {"init_method", (getter)KMeans_get_init_method, (setter)KMeans_set_init_method, "Initialization method", NULL},
    {"dist_method", (getter)KMeans_get_dist_method, (setter)KMeans_set_dist_method, "Distance method", NULL},
    {"centroids",   (getter)KMeans_get_centroids,   (setter)KMeans_set_centroids,   "Cluster centroids", NULL},
    {"labels",      (getter)KMeans_get_labels,      (setter)KMeans_set_labels,      "Cluster labels", NULL},
    {"distances",   (getter)KMeans_get_distances,   (setter)KMeans_set_distances,   "Distances", NULL},
    {"mask",        (getter)KMeans_get_mask,        (setter)KMeans_set_mask,        "Mask", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject KMeansType =
{
    .ob_base      = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "KMeans",
    .tp_basicsize = sizeof(m00nny::KMeans),
    .tp_dealloc   = (destructor)KMeans_dealloc,
    .tp_repr      = (reprfunc)KMeans_repr,
    .tp_call      = (ternaryfunc)KMeans_call,
    .tp_str       = (reprfunc)KMeans_str,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "KMeans object",
    .tp_methods   = KMeans_methods,
    .tp_members   = KMeans_members,
    .tp_getset    = KMeans_getsetters,
    .tp_init      = (initproc)KMeans_init,
    .tp_alloc     = (allocfunc)KMeans_alloc,
    .tp_new       = (newfunc)KMeans_new,
    .tp_free      = (freefunc)KMeans_free,
};

#define KMeansObject_Check(v) Py_IS_TYPE(v, &KMeansType)
#endif // LIBM00NNY_KMEANS_H