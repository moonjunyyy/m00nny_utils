#ifndef LIBM00NNY_TSNE_H
#define LIBM00NNY_TSNE_H

#include "linalg/linalg.h"
#include "distance/distance.h"
#include "algorithm/barnes_hut.h"
#include "common/common.h"
#include "common/macros.h"
#include "common/pyarg.h"

#include <utility>

namespace m00nny {

/*
t-SNE (van der Maaten & Hinton 2008) with second-order steps.

Objective: KL(P || Q) over the embedding Y (N x d), P from the input with per-point Gaussian bandwidths matched to
the perplexity (binary search), symmetrized; Q from the Student-t kernel w_ij = 1 / (1 + |y_i - y_j|^2).

Step: the direction solves  B dY = -grad  with the partial Hessian  B = 4 a L(P) (x) I_d + mu I  (spectral
direction, Vladymyrov & Carreira-Perpinan 2012): L(P) is the graph Laplacian of P and a the exaggeration factor.
4 a L(P) is the Hessian of the attractive term  sum a p_ij log(1 + |y_i - y_j|^2)  at Y = 0 and bounds it from
above everywhere (per pair, the Hessian of log(1 + |u|^2) is at most 2 I), it is positive semidefinite and does not
change between iterations, so it is factored once per phase; mu is a small shift that makes it definite. The step
length comes from a backtracking (Armijo) line search starting at learning_rate.

exact: dense P and exact repulsion, O(N^2) per evaluation, B by Cholesky.
barnes_hut: P on the 3 * perplexity nearest neighbours (sparse), repulsion by a Barnes-Hut 2^d-tree with the
given angle, O(N log N) per evaluation, B by preconditioned conjugate gradients; n_components <= 3.
*/
class TSNE {
public:    
    PyObject_HEAD;

    enum InitMethod : int8_t { PCA, Random };
    enum Method : int8_t { Exact, BarnesHut };
    static constexpr std::pair<const char*, int8_t> kInitMethods[] = {{"pca", PCA}, {"random", Random}};
    static constexpr std::pair<const char*, int8_t> kMethods[]     = {{"exact", Exact}, {"barnes_hut", BarnesHut}};

    // Defaults, shared by the Python binding and algorithm/tsne.pyi.
    static constexpr int64_t kDefaultNComponents          = 2;
    static constexpr double  kDefaultPerplexity           = 30.0;
    static constexpr double  kDefaultLearningRate         = 1.0;    // first trial step, in units of the second-order step
    static constexpr double  kDefaultEarlyExaggeration    = 12.0;
    static constexpr int64_t kDefaultMaxIter              = 1000;
    static constexpr int64_t kDefaultNIterWithoutProgress = 300;
    static constexpr int64_t kDefaultBatchSize            = 1024;
    static constexpr int8_t  kDefaultInitMethod           = PCA;
    static constexpr int8_t  kDefaultDistMethod           = Euclidean;
    static constexpr int8_t  kDefaultMethod               = BarnesHut;
    static constexpr double  kDefaultAngle                = 0.5;

    int64_t n_components;
    double  perplexity;
    double  learning_rate;            // first trial step of the line search
    double  early_exaggeration;       // P is multiplied by this in the first phase
    int64_t max_iter;                 // iterations in all (the exaggeration phase included)
    int64_t n_iter_without_progress;  // stop after this many iterations without a decrease of the objective
    int64_t batch_size;               // rows per batch of input distances (an upper bound; memory caps it too)
    int8_t  init_method;
    int8_t  dist_method;              // input-space metric (euclidean distances enter P squared)
    int8_t  tsne_method;
    double  angle;                    // Barnes-Hut accuracy: 0 is exact
    int64_t random_state;

    double  kl_divergence;            // after fit_transform: KL(P || Q) of the result
    int64_t n_iter;                   // after fit_transform: iterations run

    // X: (N, D). Returns the embedding (N, n_components) in X's dtype, on X's device.
    torch::Tensor fit_transform(const torch::Tensor& X);
};
} // namespace m00nny

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#define PY_SSIZE_T_CLEAN

/*
Python C API Functions
*/

PyObject* TSNE_alloc            (PyTypeObject *type, Py_ssize_t nitems=0);                  // Allocate memory for the class (C++ operations)
PyObject* TSNE_new              (PyTypeObject *type, PyObject *args, PyObject *kwds);       // Create a new object (Python operations)
int       TSNE_init             (m00nny::TSNE *self, PyObject *args, PyObject *kwds);       // Initialize the Instance (Python operations)
void      TSNE_free             (m00nny::TSNE *self);                                       // Free the memory for the object (Python operations)
void      TSNE_dealloc          (m00nny::TSNE *self);                                       // Deallocate memory for the class (C++ operations)
PyObject* TSNE_call             (m00nny::TSNE *self, PyObject *args, PyObject *kwds);       // Function when the object is called
PyObject* TSNE_str              (m00nny::TSNE *self);                                       // Function when the object is converted to string
PyObject* TSNE_repr             (m00nny::TSNE *self);                                       // Function when the object is printed

/*
C++ Class Wrapper Functions
*/
PyObject* TSNE_fit_transform (m00nny::TSNE *self, PyObject *args, PyObject *kwds);

/*
C++ Class Getters / Setters of the named settings
*/
PyObject* TSNE_get_init_method (PyObject *self, void* closure);
PyObject* TSNE_get_dist_method (PyObject *self, void* closure);
PyObject* TSNE_get_tsne_method (PyObject *self, void* closure);
int TSNE_set_init_method (PyObject *self, PyObject *value, void* closure);
int TSNE_set_dist_method (PyObject *self, PyObject *value, void* closure);
int TSNE_set_tsne_method (PyObject *self, PyObject *value, void* closure);

#ifdef __cplusplus
}
#endif // __cplusplus

static PyMethodDef TSNE_methods[] =
{
    {"fit_transform", (PyCFunction)TSNE_fit_transform, METH_VARARGS | METH_KEYWORDS, "Fit the data and transform it."},
    {NULL, NULL, 0, NULL}
};

static PyMemberDef TSNE_members[] =
{
    {"n_components",            Py_T_LONGLONG, offsetof(m00nny::TSNE, n_components),            0, "Dimension of the embedding."},
    {"perplexity",              Py_T_DOUBLE,   offsetof(m00nny::TSNE, perplexity),              0, "Perplexity (effective number of neighbours)."},
    {"learning_rate",           Py_T_DOUBLE,   offsetof(m00nny::TSNE, learning_rate),           0, "First trial step of the line search, in units of the second-order step."},
    {"early_exaggeration",      Py_T_DOUBLE,   offsetof(m00nny::TSNE, early_exaggeration),      0, "Factor on P in the first phase."},
    {"max_iter",                Py_T_LONGLONG, offsetof(m00nny::TSNE, max_iter),                0, "Iterations in all."},
    {"n_iter_without_progress", Py_T_LONGLONG, offsetof(m00nny::TSNE, n_iter_without_progress), 0, "Stop after this many iterations without progress."},
    {"batch_size",              Py_T_LONGLONG, offsetof(m00nny::TSNE, batch_size),              0, "Rows per batch of input distances."},
    {"angle",                   Py_T_DOUBLE,   offsetof(m00nny::TSNE, angle),                   0, "Barnes-Hut accuracy (0: exact)."},
    {"random_state",            Py_T_LONGLONG, offsetof(m00nny::TSNE, random_state),            0, "Seed of the random initialization."},
    {"kl_divergence",           Py_T_DOUBLE,   offsetof(m00nny::TSNE, kl_divergence),           Py_READONLY, "KL(P || Q) after fit_transform."},
    {"n_iter",                  Py_T_LONGLONG, offsetof(m00nny::TSNE, n_iter),                  Py_READONLY, "Iterations run by fit_transform."},
    {NULL}
};

static PyGetSetDef TSNE_getsetters[] =
{
    {"init_method", (getter)TSNE_get_init_method, (setter)TSNE_set_init_method, "Initialization: 'pca' or 'random'.", NULL},
    {"dist_method", (getter)TSNE_get_dist_method, (setter)TSNE_set_dist_method, "Input metric: 'euclidean', 'cosine', 'manhattan' or p.", NULL},
    {"tsne_method", (getter)TSNE_get_tsne_method, (setter)TSNE_set_tsne_method, "'exact' or 'barnes_hut'.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject TSNEType =
{
    .ob_base      = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "TSNE",
    .tp_basicsize = sizeof(m00nny::TSNE),
    .tp_dealloc   = (destructor)TSNE_dealloc,
    .tp_repr      = (reprfunc)TSNE_repr,
    .tp_call      = (ternaryfunc)TSNE_call,
    .tp_str       = (reprfunc)TSNE_str,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "TSNE object",
    .tp_methods   = TSNE_methods,
    .tp_members   = TSNE_members,
    .tp_getset    = TSNE_getsetters,
    .tp_init      = (initproc)TSNE_init,
    .tp_alloc     = (allocfunc)TSNE_alloc,
    .tp_new       = (newfunc)TSNE_new,
};

#define TSNEObject_Check(v) Py_IS_TYPE(v, &TSNEType)
#endif // LIBM00NNY_TSNE_H
