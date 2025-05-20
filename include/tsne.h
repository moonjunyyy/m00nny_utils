#ifndef LIBM00NNY_TSNE_H
#define LIBM00NNY_TSNE_H

#include "linalg.h"
#include "distance.h"
#include "common/common.h"
#include "common/macros.h"

namespace m00nny {

class BarnesHutTree {
    public:
    torch::Tensor data;
    torch::Tensor state;
    torch::Tensor data_center;
    torch::Tensor state_center;
    int64_t size;
    int64_t depth;
    int64_t data_dimension;
    int64_t state_dimension;
    int64_t batch_size;
    double angle;

    std::shared_ptr<BarnesHutTree> root;
    std::shared_ptr<BarnesHutTree> parent;
    std::vector<std::shared_ptr<BarnesHutTree>> children;
    std::vector<std::shared_ptr<BarnesHutTree>> leaves;

    BarnesHutTree
    (torch::Tensor& X,
     torch::Tensor& S,
     int64_t batch_size,
     double angle=0.5,
     int64_t depth=0,
     std::shared_ptr<BarnesHutTree> parent=nullptr)
        : size(S.size(0)),
          batch_size(batch_size),
          depth(depth),
          data_dimension(X.size(1)),
          state_dimension(S.size(1)),
          angle(angle)
    {
        if (S.size(0) != X.size(0)) { printf("Error: The size of the data and the state must be the same.\n"); return; }
        this->data = X;  // The data is not normalized, we don't need to copy the memory.
        this->state = S; // The state is not normalized, we don't need to copy the memory.
        this->parent = parent;
        if (this->parent && depth != 0) this->root = this->parent->root;
        else
        {
            this->root = std::make_shared<BarnesHutTree>(*this);
            this->state.sub_(S.mean(0, true)).div_(S.std(0, true));
        }

        this->state_center = S.mean(0, true);
        torch::Tensor diff = S - this->state_center;
        double dist = (this->data - this->state_center).norm().item<double>();
        if ((this->size / dist) < this->angle)
        {
            this->data_center = X.mean(0, true);
            this->root->leaves.push_back(std::make_shared<BarnesHutTree>(*this));
        }
        else
        {
            int64_t _width = 1 << this->state_dimension;
            this->children = std::vector<std::shared_ptr<BarnesHutTree>>(_width);
            torch::Tensor mask = torch::zeros({this->size}, torch::kInt64);
            for (int64_t j = 0; j < this->state_dimension; j++){
                mask.add_(this->data.select(1, j).lt(this->state_center.select(1, j)).to(torch::kInt64).bitwise_left_shift(j));}
            for (int64_t i = 0; i < _width; i++){
                torch::Tensor indices = mask.eq(i).nonzero().squeeze();
                torch::Tensor _data = this->data.index_select(0, indices);
                torch::Tensor _state = this->state.index_select(0, indices);
                this->children[i] = std::make_shared<BarnesHutTree>(
                    _data, _state, this->batch_size, this->angle, this->depth + 1,
                    std::make_shared<BarnesHutTree>(*this));}
        }
    }
    std::tuple<torch::Tensor, torch::Tensor> get_leaves()
    {
        torch::Tensor leaves_data;
        torch::Tensor leaves_state;
        for (int64_t i = 0; i < (int64_t)this->root->leaves.size(); i++)
        {
            leaves_data[i] = this->root->leaves[i]->data_center;
            leaves_state[i] = this->root->leaves[i]->state_center;
        }
        return std::make_tuple(leaves_data, leaves_state);
    }

    BarnesHutTree& operator-= (const torch::Tensor& other)
    {
        for (int64_t i = 0; i < (int64_t)this->leaves.size(); i++)
            if (this->leaves[i]) this->leaves[i]->state -= other.slice(0, i, i+1);
        return *this;
    }
};

class TSNE {
public:    
    PyObject_HEAD;

    enum TsneInitMethod { PCA, Random };
    enum TsneMethod { Exact, BarnesHut };

    int64_t n_components;
    double  perplexity;
    double  early_exaggeration;
    double  learning_rate;
    int64_t max_iter;
    int64_t n_iter_without_progress;
    int64_t batch_size;
    int8_t  init_method;
    int8_t  dist_method;
    int8_t  tsne_method;
    double  angle;
    int64_t random_state;

    struct TSNEState
    {
        /*
        Data and Probabilities.
        */
        torch::Tensor X;  // Data.
        torch::Tensor S;  // State.
        // !!! This is a copy of the state, not a view. Not shares the memory with the state.!!!
        torch::Tensor St; // Transpose of the state.
        torch::Tensor P;  // Joint probabilities for the data.
        torch::Tensor Q;  // Joint probabilities for the state.

        /*
        Derivatives.
        */
        torch::Tensor dDs_dSt;    // Reserved memory for Derivative of Ds w.r.t. S.
        torch::Tensor dDs_dS;     // transpose of dDs_dSt for the calculation.
        torch::Tensor dKL_dSt;    // Reserved memory for Derivative of KL w.r.t. St.
        torch::Tensor dKL_dS;     // transpose of dKL_dSt for the calculation.
        torch::Tensor dKL_dDs;    // Derivative of KL w.r.t. Ds.
        torch::Tensor ddKL_dDsDs; // Second derivative of KL w.r.t. Ds.
        torch::Tensor ddKL_dSdSt; // Second derivative of KL w.r.t. S.

        /*
        Calculation buffers. 
        */
        torch::Tensor Dd; // Distance matrix for the data.
        torch::Tensor Ds; // Distance matrix for the state.
        torch::Tensor normD; // Norm of the data.
        torch::Tensor normS; // Norm of the state.
        torch::Tensor diffD; // Difference of the data.
        torch::Tensor diffS; // Difference of the state.
        torch::Tensor buffer; // Buffer for the etc.
    };

    TSNE(int64_t n_components,
         double  perplexity,
         double  learning_rate,
         double  early_exaggeration,
         int64_t max_iter,
         int64_t n_iter_without_progress,
         int64_t batch_size,
         int8_t  init_method,
         int8_t  dist_method,
         int8_t  tsne_method,
         double  angle,
         int64_t random_state
        );

    template <typename _T>
    torch::Tensor& fit_transform(_T&& X);

    const char* init_method_str();
    const char* dist_method_str();
    const char* tsne_method_str();

private:
    std::unique_ptr<TSNEState> init_state(torch::Tensor& X);
    std::unique_ptr<TSNEState> init_state(torch::Tensor& X, torch::Tensor& S, TSNEState* state);
    void exact(TSNEState* state, double perplexity);
    void barnes_hut(TSNEState* state, double perplexity);
    void cosine(TSNEState* state);
    void pnorm(TSNEState* state, int64_t p);
    void joint_probabilities(TSNEState* state, float perplexity);
};
} // namespace m00nny

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#define PY_SSIZE_T_CLEAN

/*
Python C API Functions
*/

PyObject* TSNE_alloc            (PyTypeObject *type, Py_ssize_t nitems=0);            // Allocate memory for the class (C++ operations)
PyObject* TSNE_new              (PyTypeObject *type, PyObject *args, PyObject *kwds); // Create a new object (Python operations)
int       TSNE_init             (m00nny::TSNE *self, PyObject *args, PyObject *kwds);       // Initialize the Instance (Python operations)
void      TSNE_finalize         (m00nny::TSNE *self);                                       // Finalize the Instance (Python operations)
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
C++ Class Getters
*/
PyObject* TSNE_get_init_method (PyObject *self, void* closure);
PyObject* TSNE_get_dist_method (PyObject *self, void* closure);
PyObject* TSNE_get_tsne_method (PyObject *self, void* closure);

/*
C++ Class Setters
*/
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
    {"n_components", Py_T_INT, offsetof(m00nny::TSNE, n_components), 0, "Number of components."},
    {"perplexity",   Py_T_DOUBLE, offsetof(m00nny::TSNE, perplexity), 0, "Perplexity."},
    {"learning_rate", Py_T_DOUBLE, offsetof(m00nny::TSNE, learning_rate), 0, "Learning rate."},
    {"early_exaggeration", Py_T_DOUBLE, offsetof(m00nny::TSNE, early_exaggeration), 0, "Early exaggeration."},
    {"max_iter", Py_T_INT, offsetof(m00nny::TSNE, max_iter), 0, "Maximum number of iterations."},
    {"n_iter_without_progress", Py_T_INT, offsetof(m00nny::TSNE, n_iter_without_progress), 0, "Number of iterations without progress."},
    {"batch_size", Py_T_INT, offsetof(m00nny::TSNE, batch_size), 0, "Batch size."},
    {"angle", Py_T_DOUBLE, offsetof(m00nny::TSNE, angle), 0, "Angle for the Barnes-Hut approximation."},
    {"random_state", Py_T_INT, offsetof(m00nny::TSNE, random_state), 0, "Random state."},
    {NULL}
};

static PyGetSetDef TSNE_getsetters[] =
{
    {"init_method", (getter)TSNE_get_init_method, (setter)TSNE_set_init_method, "Initialization method.", NULL},
    {"dist_method", (getter)TSNE_get_dist_method, (setter)TSNE_set_dist_method, "Distance method.", NULL},
    {"tsne_method", (getter)TSNE_get_tsne_method, (setter)TSNE_set_tsne_method, "TSNE method.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject TSNEType =
{
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "TSNE",
    .tp_basicsize = sizeof(m00nny::TSNE),
    .tp_dealloc   = (destructor)TSNE_dealloc,
    .tp_repr      = (reprfunc)TSNE_repr,
    .tp_call      = (ternaryfunc)TSNE_call,
    .tp_str       = (reprfunc)TSNE_str,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "TSNE object",
    .tp_methods   = TSNE_methods,
    .tp_getset    = TSNE_getsetters,
    .tp_init      = (initproc)TSNE_init,
    .tp_alloc     = (allocfunc)TSNE_alloc,
    .tp_new       = (newfunc)TSNE_new,
    .tp_free      = (freefunc)TSNE_free,
};

#define TSNEObject_Check(v) Py_IS_TYPE(v, &TSNEType)
#endif // LIBM00NNY_TSNE_H