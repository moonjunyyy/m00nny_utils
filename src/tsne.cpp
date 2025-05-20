#include "tsne.h"

/*
C++ class implementation
*/
m00nny::TSNE::TSNE
(int64_t n_components,
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
 int64_t random_state)
{
    this->n_components           = n_components;
    this->perplexity             = perplexity;
    this->learning_rate          = learning_rate;
    this->early_exaggeration     = early_exaggeration;
    this->max_iter               = max_iter;
    this->n_iter_without_progress = n_iter_without_progress;
    this->batch_size             = batch_size;
    this->init_method            = init_method;
    this->dist_method            = dist_method;
    this->tsne_method            = tsne_method;
    this->angle                  = angle;
    this->random_state           = random_state;
}

template <typename _T>
torch::Tensor&
m00nny::TSNE::fit_transform
(_T&& X)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::fit_transform\n")
    torch::NoGradGuard no_grad;
    auto _X = std::forward<_T>(X);
    std::unique_ptr<TSNEState> state = std::move(this->init_state(_X));
    
    double perplexity = this->perplexity;
    double error = 0;
    int64_t n_iter_without_progress = 0;
    torch::Tensor old_S = torch::empty({state->S.size(0), state->S.size(1)}, state->S.options());
    for (int64_t i = 0; i < this->max_iter; i++)
    {
        if (i < this->early_exaggeration) perplexity *= 4;
        else if (i == this->early_exaggeration) perplexity /= 4;

        if (this->tsne_method == Exact) 
            this->exact(state.get(), perplexity);
        else if (this->tsne_method == BarnesHut) 
            this->barnes_hut(state.get(), perplexity);
        else { printf("Error: Unknown tsne_method\n"); return state->S; }

        error = torch::norm(state->S - old_S, 2).item<double>();
        if (error < 1e-4) n_iter_without_progress++;
        else n_iter_without_progress = 0;
        if (n_iter_without_progress >= this->n_iter_without_progress) break;
        old_S.copy_(state->S);
    }
    return state->S;
}

void
m00nny::TSNE::exact
(m00nny::TSNE::TSNEState* state, double perplexity)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::exact\n")
    torch::NoGradGuard no_grad;
    if (this->dist_method < 0)
        {printf("Error: Unknown dist_method\n"); return;}
    else if (this->dist_method == Cosine) 
        this->cosine(state);
    else 
        this->pnorm(state, this->dist_method);
    this->joint_probabilities(state, perplexity);
    state->S -= torch::matmul(m00nny::inverse(state->ddKL_dSdSt), state->dKL_dS);
}

void
m00nny::TSNE::barnes_hut
(m00nny::TSNE::TSNEState* state, double perplexity)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::barnes_hut\n")
    torch::NoGradGuard no_grad;
    BarnesHutTree tree = BarnesHutTree(state->X, state->S, this->batch_size, this->angle);
    auto [leaves_data, leaves_state] = tree.get_leaves();

    std::unique_ptr<TSNEState> substate = std::move(this->init_state(leaves_data, leaves_state, state));
    if (this->dist_method < 0)
        {printf("Error: Unknown dist_method\n"); return;}
    else if (this->dist_method == Cosine) 
        this->cosine(substate.get());
    else 
        this->pnorm(substate.get(), this->dist_method);
    this->joint_probabilities(substate.get(), perplexity);
    tree -= torch::matmul(m00nny::inverse(substate->ddKL_dSdSt), substate->dKL_dS);
}

void 
m00nny::TSNE::cosine
(TSNEState* state)
{
    torch::norm_out(state->normD, state->X, 2, 1);
    torch::matmul_out(state->Dd, state->X, state->X.t());
    torch::Tensor _buffer = state->buffer.view({state->X.size(0), state->X.size(0)});
    torch::mul_out(_buffer, state->normD.unsqueeze(1), state->normD.unsqueeze(0));
    state->Dd.div_(_buffer.add_(1e-8)).mul_(-1.).add_(1.).clamp_(0., 2.);

    torch::norm_out(state->normS, state->S, 2, 1);
    torch::matmul_out(state->Ds, state->S, state->S.t());
    torch::mul_out(_buffer, state->normS.unsqueeze(1), state->normS.unsqueeze(0));
    state->Ds.div_(_buffer.add_(1e-8)).mul_(-1.).add_(1.).clamp_(0., 2.);

    torch::matmul_out(state->dDs_dS, state->Ds, state->S);
    state->dDs_dS.mul_(-1).add_(state->S).mul_(2).div_(state->normS.unsqueeze(-1)).div_(state->normS.unsqueeze(-1));
}

void
m00nny::TSNE::pnorm
(TSNEState* state, int64_t p)
{
    torch::sub_out(state->diffD, state->X.unsqueeze(0), state->X.unsqueeze(1));
    torch::Tensor _buffer = state->buffer.view({state->X.size(0), state->X.size(0), state->X.size(1)});
    torch::abs_out(_buffer, state->diffD).pow_(p);
    torch::sum_out(state->Dd, _buffer, -1);
    state->Dd.pow_(1.0 / p);

    torch::sub_out(state->diffS, state->S.unsqueeze(0), state->S.unsqueeze(1));
    torch::Tensor _buffer_absS  = _buffer.slice(-1, 0, state->X.size(1));
    torch::Tensor _buffer_absSp = _buffer.slice(-1, state->X.size(1), state->X.size(1) * 2);
    torch::Tensor _buffer_Dsp   = _buffer.slice(-1, state->X.size(1) * 2, state->X.size(1) * 2 + 1);
    torch::abs_out(_buffer_absS,  state->diffS);
    torch::pow_out(_buffer_absSp, _buffer_absS, p);
    torch::sum_out(state->Ds, _buffer_absSp, -1);

    torch::pow_out(_buffer_Dsp,   _buffer_absS, p - 1);
    torch::pow_out(_buffer_absSp, _buffer_absS, p - 2).mul_(state->diffS).div_(_buffer_Dsp.unsqueeze(-1));
    torch::sum_out(state->dDs_dS, _buffer_absSp, -1);
}

void
m00nny::TSNE::joint_probabilities
(TSNEState* state, float perplexity)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::joint_probabilities\n")

    // Pre-allocate the memory for the buffers.
    torch::Tensor _buffer_1 = state->buffer.slice(
        0, 0, state->X.size(0) * state->X.size(0))
        .view({state->X.size(0), state->X.size(0)});
    torch::Tensor _buffer_2 = state->buffer.slice(
        0, state->X.size(0) * state->X.size(0),
        state->X.size(0) * state->X.size(0) + state->X.size(0));

    // Gaussian Joint Probabilities
    torch::pow_out(_buffer_1, state->Dd, 2);
    torch::sum_out(_buffer_2, _buffer_1, 1);
    torch::div_out(state->P, state->Dd, _buffer_2.unsqueeze(1)).div_(-2).div_(perplexity).div_(perplexity).exp_();
    torch::sum_out(_buffer_2, state->P, 1);
    state->P.div_(_buffer_2.unsqueeze(1));

    // Student-t Joint Probabilities
    torch::div_out(_buffer_1, state->Ds, perplexity).add_(1).reciprocal_();
    torch::sum_out(_buffer_2, _buffer_1, 1);
    torch::div_out(state->Q, _buffer_1, _buffer_2.unsqueeze(1));

    torch::sub_out(state->dKL_dDs, state->Q, 1.0);
    state->dKL_dDs.mul_(-1).mul_(_buffer_1).div_(perplexity).mul_(state->P).div_(state->Q);

    torch::mul_out(state->ddKL_dSdSt, state->dKL_dDs, _buffer_1);
    torch::reciprocal_out(_buffer_1, state->Q).add_(1).div_(-perplexity);
    state->ddKL_dDsDs.mul_(_buffer_1);

    torch::matmul_out(state->dKL_dS, state->dKL_dDs, state->dDs_dS);
    _buffer_1 = _buffer_1.view({-1})
        .slice(0, 0, state->X.size(0) * this->n_components)
        .view({state->X.size(0), this->n_components});
    torch::matmul_out(_buffer_1, state->ddKL_dDsDs, state->dDs_dS);
    torch::matmul_out(state->ddKL_dSdSt, _buffer_1, state->dDs_dS);
}

std::unique_ptr<m00nny::TSNE::TSNEState>
m00nny::TSNE::init_state
(torch::Tensor& X)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::init_state\n")
    torch::NoGradGuard no_grad;
    std::unique_ptr<TSNEState> state = std::make_unique<TSNEState>();

    state->X = X.clone();
    // state->S  = torch::empty({X.size(0), this->n_components}, X.options()).view({X.size(0), X.size(1)});
    // state->St = torch::empty({this->n_components, X.size(0)}, X.options()).view({X.size(0), X.size(1)});
    state->P  = torch::empty({X.size(0) * X.size(0)}, X.options()).view({X.size(0), X.size(0)});
    state->Q  = torch::empty({X.size(0) * X.size(0)}, X.options()).view({X.size(0), X.size(0)});
    
    state->dDs_dSt = torch::empty({this->n_components, X.size(0)}, X.options()).view({this->n_components, X.size(0)});
    state->dKL_dSt = torch::empty({this->n_components, X.size(0)}, X.options()).view({this->n_components, X.size(0)});
    state->dDs_dS = state->dDs_dSt.t();
    state->dKL_dS = state->dKL_dSt.t();
    state->dKL_dDs = torch::empty({X.size(0) * X.size(0)}, X.options()).view({X.size(0), X.size(0)});
    state->ddKL_dDsDs = torch::empty({X.size(0) * X.size(0)}, X.options()).view({X.size(0), X.size(0)});
    state->ddKL_dSdSt = torch::empty({X.size(0) * X.size(0)}, X.options()).view({X.size(0), X.size(0)});

    state->Dd = torch::empty({X.size(0) * X.size(0)}, X.options()).view({X.size(0), X.size(0)});
    state->Ds = torch::empty({X.size(0) * X.size(0)}, X.options()).view({X.size(0), X.size(0)});
    if (this->dist_method == Cosine)
    {
        state->normD = torch::empty({X.size(0)}, X.options());
        state->normS = torch::empty({X.size(0)}, X.options());
        state->buffer = torch::empty({X.size(0) * X.size(0)}, X.options()).view({X.size(0), X.size(1)});
    } else
    {
        state->diffD = torch::empty({X.size(0) * X.size(0) * X.size(1)}, X.options()).view({X.size(0), X.size(0), X.size(1)});
        state->diffS = torch::empty({X.size(0) * X.size(0) * this->n_components}, X.options()).view({X.size(0), X.size(0), this->n_components});
        state->buffer = torch::empty({X.size(0) * X.size(0) * X.size(1)}, X.options()).view({X.size(0), X.size(0), X.size(1)});
    }
    
    torch::sub_out(state->X, X, torch::mean(X, 0, true));
    if (this->init_method == PCA)
    {
        // PCA Initialization
        torch::Tensor U, Σ, V;
        std::tie(U, Σ, V) = singular_value_decomposition(state->X, this->n_components, 1000);
        state->S = state->X.mul(U.slice(1, 0, this->n_components));
    }
    else if (this->init_method == Random)
    {
        // Random Initialization
        state->S = torch::randn({X.size(0), this->n_components});
    }
    else
    {
        printf("Error: Unknown init_method\n");
        return nullptr;
    }
    state->St = state->S.t().clone();
    return state;
}


std::unique_ptr<m00nny::TSNE::TSNEState>
m00nny::TSNE::init_state
(torch::Tensor& X, torch::Tensor& S, m00nny::TSNE::TSNEState* original_state)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::init_state\n")
    torch::NoGradGuard no_grad;
    std::unique_ptr<TSNEState> state = std::make_unique<TSNEState>();

    state->X = X;
    state->S = S;
    state->St = S.t().clone();
    state->P = original_state->P.view({-1}).slice(0, 0, X.size(0) * X.size(0)).view({X.size(0), X.size(0)});
    state->Q = original_state->Q.view({-1}).slice(0, 0, X.size(0) * X.size(0)).view({X.size(0), X.size(0)});

    state->dDs_dSt = original_state->dDs_dSt.view({-1}).slice(0, 0, this->n_components * X.size(0)).view({this->n_components, X.size(0)});
    state->dKL_dSt = original_state->dKL_dSt.view({-1}).slice(0, 0, this->n_components * X.size(0)).view({this->n_components, X.size(0)});
    state->dDs_dS = state->dDs_dSt.t();
    state->dKL_dS = state->dKL_dSt.t();
    state->dKL_dDs = original_state->dKL_dDs.view({-1}).slice(0, 0, X.size(0) * X.size(0)).view({X.size(0), X.size(0)});
    state->ddKL_dDsDs = original_state->ddKL_dDsDs.view({-1}).slice(0, 0, X.size(0) * X.size(0)).view({X.size(0), X.size(0)});
    state->ddKL_dSdSt = original_state->ddKL_dSdSt.view({-1}).slice(0, 0, X.size(0) * X.size(0)).view({X.size(0), X.size(0)});

    state->Dd = original_state->Dd.view({-1}).slice(0, 0, X.size(0) * X.size(0)).view({X.size(0), X.size(0)});
    state->Ds = original_state->Ds.view({-1}).slice(0, 0, X.size(0) * X.size(0)).view({X.size(0), X.size(0)});
    if (this->dist_method == Cosine)
    {
        state->normD  = original_state->normD.slice(0, 0, X.size(0));
        state->normS  = original_state->normS.slice(0, 0, X.size(0));
        state->buffer = original_state->buffer.view({-1}).slice(0, 0, X.size(0) * X.size(0)).view({X.size(0), X.size(0)});
    }
    else
    {
        state->diffD  = original_state->diffD.view({-1}).slice(0, 0, X.size(0) * X.size(0) * X.size(1)).view({X.size(0), X.size(0), X.size(1)});
        state->diffS  = original_state->diffS.view({-1}).slice(0, 0, X.size(0) * X.size(0) * this->n_components).view({X.size(0), X.size(0), this->n_components});
        state->buffer = original_state->buffer.view({-1}).slice(0, 0, X.size(0) * X.size(0) * X.size(1)).view({X.size(0), X.size(0), X.size(1)});
    }
    return state;
}

const char* 
m00nny::TSNE::init_method_str()
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::init_method_str\n")
    switch (this->init_method)
    {
        case PCA:    return "pca";
        case Random: return "random";
        default:     return "unknown";
    }
}
const char* 
m00nny::TSNE::dist_method_str()
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::dist_method_str\n")
    return metric_str(this->dist_method);
}
const char* 
m00nny::TSNE::tsne_method_str()
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::tsne_method_str\n")
    switch (this->tsne_method)
    {
        case Exact:    return "exact";
        case BarnesHut: return "barnes_hut";
        default:        return "unknown";
    }
}

/*
Python C API Functions
*/

PyObject* 
TSNE_alloc
(PyTypeObject *type, Py_ssize_t nitems)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__alloc__\n")
    // Allocate memory for the class.
    PyObject* ret;
    InitGILScope
    ret = PyType_GenericAlloc(type, nitems);
    if (ret == NULL) return NULL;
    ExitGILScope
    return ret;
}

PyObject* 
TSNE_new
(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__new__\n")
    // Generate a new instance of the class.
    PyObject* self;
    InitGILScope
    self = PyType_GenericNew(type, args, kwds);
    ExitGILScope
    return self;
}

int 
TSNE_init 
(m00nny::TSNE *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__init__\n")
    // Initialize the instance.

    PyObject * n_components, * perplexity, * learning_rate, 
             * early_exaggeration, * max_iter, * n_iter_without_progress,
             * batch_size, * init_method, * dist_method,
             * tsne_method, * angle, * random_state;

    const char* kwlist[] = {
        "n_components", "perplexity", "learning_rate", "early_exaggeration",
        "max_iter", "n_iter_without_progress", "batch_size", "init_method",
        "dist_method", "tsne_method", "angle", "random_state", NULL};

    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOOOOOOOOOO", (char**)kwlist,
        &n_components, &perplexity, &learning_rate, &early_exaggeration,
        &max_iter, &n_iter_without_progress, &batch_size, &init_method,
        &dist_method, &tsne_method, &angle, &random_state)) return -1;

    int64_t _stack = 0;
    IfSaveState(_stack, (n_components==nullptr) || Py_IsNone(n_components), [](){ PyErr_SetString (PyExc_TypeError, "n_clusters is required.");})
    IfSaveState(_stack, (perplexity==nullptr) || Py_IsNone(perplexity), [&perplexity](){perplexity = PyFloat_FromDouble(30.0);})
    IfSaveState(_stack, (learning_rate==nullptr) || Py_IsNone(learning_rate), [&learning_rate](){learning_rate = PyFloat_FromDouble(200.0);})
    IfSaveState(_stack, (early_exaggeration==nullptr) || Py_IsNone(early_exaggeration), [&early_exaggeration](){early_exaggeration = PyFloat_FromDouble(12.0);})
    IfSaveState(_stack, (max_iter==nullptr) || Py_IsNone(max_iter), [&max_iter](){max_iter = PyLong_FromLong(1000);})
    IfSaveState(_stack, (n_iter_without_progress==nullptr) || Py_IsNone(n_iter_without_progress), [&n_iter_without_progress](){n_iter_without_progress = PyLong_FromLong(300);})
    IfSaveState(_stack, (batch_size==nullptr) || Py_IsNone(batch_size), [&batch_size](){batch_size = PyLong_FromLong(1024);})
    IfSaveState(_stack, (init_method==nullptr) || Py_IsNone(init_method), [&init_method](){init_method = PyUnicode_FromString("pca");})
    IfSaveState(_stack, (dist_method==nullptr) || Py_IsNone(dist_method), [&dist_method](){dist_method = PyUnicode_FromString("euclidean");})
    IfSaveState(_stack, (tsne_method==nullptr) || Py_IsNone(tsne_method), [&tsne_method](){tsne_method = PyUnicode_FromString("barnes_hut");})
    IfSaveState(_stack, (angle==nullptr) || Py_IsNone(angle), [&angle](){angle = PyFloat_FromDouble(0.5);})
    IfSaveState(_stack, (random_state==nullptr) || Py_IsNone(random_state), [&random_state](){random_state = PyLong_FromLong([](){ std::random_device rd; std::mt19937 gen(rd()); return gen(); }()); })

    // We have strong references to the objects, so we don't need to Py_INCREF.
    PyObject_GenericSetAttrWithString((PyObject*)self, "n_components",           n_components);
    PyObject_GenericSetAttrWithString((PyObject*)self, "perplexity",             perplexity);
    PyObject_GenericSetAttrWithString((PyObject*)self, "learning_rate",          learning_rate);
    PyObject_GenericSetAttrWithString((PyObject*)self, "early_exaggeration",     early_exaggeration);
    PyObject_GenericSetAttrWithString((PyObject*)self, "max_iter",               max_iter);
    PyObject_GenericSetAttrWithString((PyObject*)self, "n_iter_without_progress", n_iter_without_progress);
    PyObject_GenericSetAttrWithString((PyObject*)self, "batch_size",             batch_size);
    PyObject_GenericSetAttrWithString((PyObject*)self, "init_method",            init_method);
    PyObject_GenericSetAttrWithString((PyObject*)self, "dist_method",            dist_method);
    PyObject_GenericSetAttrWithString((PyObject*)self, "tsne_method",            tsne_method);
    PyObject_GenericSetAttrWithString((PyObject*)self, "angle",                  angle);
    PyObject_GenericSetAttrWithString((PyObject*)self, "random_state",           random_state);

    IfRestoreState(_stack, [&random_state](){ Py_XDECREF(random_state); })
    IfRestoreState(_stack, [&angle](){ Py_XDECREF(angle); })
    IfRestoreState(_stack, [&tsne_method](){ Py_XDECREF(tsne_method); })
    IfRestoreState(_stack, [&dist_method](){ Py_XDECREF(dist_method); })
    IfRestoreState(_stack, [&init_method](){ Py_XDECREF(init_method); })
    IfRestoreState(_stack, [&batch_size](){ Py_XDECREF(batch_size); })
    IfRestoreState(_stack, [&n_iter_without_progress](){ Py_XDECREF(n_iter_without_progress); })
    IfRestoreState(_stack, [&max_iter](){ Py_XDECREF(max_iter); })
    IfRestoreState(_stack, [&early_exaggeration](){ Py_XDECREF(early_exaggeration); })
    IfRestoreState(_stack, [&learning_rate](){ Py_XDECREF(learning_rate); })
    IfRestoreState(_stack, [&perplexity](){ Py_XDECREF(perplexity); })
    IfRestoreState(_stack, [&n_components](){ Py_XDECREF(n_components); })
    ExitGILScope
    return 0;
}

void
TSNE_finalize
(m00nny::TSNE *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__finalize__\n")
    // Finalize the instance of the class.
    // This function is called the end of the life cycle of the instance.
    // This not means the deallocation of the memory by garbage collector.
    // In this case, nothing to do.
    return;
}

void
TSNE_free
(m00nny::TSNE *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__free__\n")
    // Free the memory for an instance of the class.
    // This function is called when the instance is deallocated.
    // If there is no shared memory between instances,
    // this function not necessaryly to do something.
    return;
}

void
TSNE_dealloc
(m00nny::TSNE *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: KMeans.__dealloc__\n")
    // Deallocating the memory for the class.
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject*
TSNE_call
(m00nny::TSNE *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__call__\n")
    // This function is called when the instance is called.
    // Get the parameter, assuming it is a tensor.
    // Call the fit_predict function.
    return TSNE_fit_transform(self, args, kwds);
}

PyObject*
TSNE_str
(m00nny::TSNE *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__str__\n")
    // Function when the object is printed
    return PyUnicode_FromFormat("<TSNE: n_components=%ld, perplexity=%f, learning_rate=%f, early_exaggeration=%f, max_iter=%ld, n_iter_without_progress=%ld, batch_size=%ld, init_method=%s, dist_method=%s, tsne_method=%s, angle=%f, random_state=%ld>",
        self->n_components, self->perplexity, self->learning_rate, self->early_exaggeration,
        self->max_iter, self->n_iter_without_progress, self->batch_size, self->init_method_str(),
        self->dist_method_str(), self->tsne_method_str(), self->angle, self->random_state);
}

PyObject*
TSNE_repr
(m00nny::TSNE *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__repr__\n")
    // Function when the object is printed
    return TSNE_str(self);
}

/*
C++ Class Wrapper Functions
*/
PyObject*
TSNE_fit_transform
(m00nny::TSNE *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.fit_transform\n")
    // This function is called when the instance is called.
    // Get the parameter, assuming it is a tensor.
    // Call the fit_predict function.
    PyObject* pyX;
    torch::Tensor X;
    const char* kwlist[] = {"X", NULL};
    
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    X = THPVariable_Unpack(pyX);
    Py_XINCREF(pyX);
    ExitGILScope

    // Call the C++ fit function.
    torch::Tensor& S = self->fit_transform(X);
    PyObject* pyS;
    InitGILScope
    pyS = THPVariable_Wrap(S);
    Py_XDECREF(pyX);
    ExitGILScope
    return pyS;
}

/*
C++ Class Getters
*/
PyObject*
TSNE_get_init_method
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.get_init_method\n")
    // Get the init_method attribute.
    PyObject* ret;
    InitGILScope
    ret = PyUnicode_FromString(((m00nny::TSNE*)self)->init_method_str());
    ExitGILScope
    return ret;
}

PyObject*
TSNE_get_dist_method
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.get_dist_method\n")
    // Get the dist_method attribute.
    PyObject* ret;
    InitGILScope
    ret = PyUnicode_FromString(((m00nny::TSNE*)self)->dist_method_str());
    ExitGILScope
    return ret;
}

PyObject*
TSNE_get_tsne_method
(PyObject *self, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.get_tsne_method\n")
    // Get the tsne_method attribute.
    PyObject* ret;
    InitGILScope
    ret = PyUnicode_FromString(((m00nny::TSNE*)self)->tsne_method_str());
    ExitGILScope
    return ret;
}

/*
C++ Class Setters
*/
int
TSNE_set_init_method
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.set_init_method\n")
    // Set the init_method attribute.
    InitGILScope
    if (PyUnicode_Check(value))
    {
        const char* value_str = PyUnicode_AsUTF8(value);
        if (strcmp(value_str, "pca") == 0)
            ((m00nny::TSNE*)self)->init_method = m00nny::TSNE::PCA;
        else if (strcmp(value_str, "random") == 0)
            ((m00nny::TSNE*)self)->init_method = m00nny::TSNE::Random;
        else { PyErr_SetString(PyExc_ValueError, "Unknown init_method"); return -1; }
    }
    else { PyErr_SetString(PyExc_TypeError, "init_method must be a string"); return -1; }
    ExitGILScope
    return 0;
}

int
TSNE_set_dist_method
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.set_dist_method\n")
    // Set the dist_method attribute.
    InitGILScope
    if (PyUnicode_Check(value))
    {
        const char* value_str = PyUnicode_AsUTF8(value);
        if (strcmp(value_str, "euclidean") == 0)
            ((m00nny::TSNE*)self)->dist_method = m00nny::Euclidean;
        else if (strcmp(value_str, "cosine") == 0)
            ((m00nny::TSNE*)self)->dist_method = m00nny::Cosine;
        else { PyErr_SetString(PyExc_ValueError, "Unknown dist_method"); return -1; }
    }
    else if (PyLong_Check(value))
    {
        int8_t value_int = PyLong_AsLong(value);
        if (value_int < 0) { PyErr_SetString(PyExc_ValueError, "dist_method must be a positive integer"); return -1; }
        ((m00nny::TSNE*)self)->dist_method = value_int;
    }
    else { PyErr_SetString(PyExc_TypeError, "dist_method must be a string or an integer"); return -1; }
    ExitGILScope
    return 0;
}

int
TSNE_set_tsne_method
(PyObject *self, PyObject *value, void* closure)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.set_tsne_method\n")
    // Set the tsne_method attribute.
    InitGILScope
    if (PyUnicode_Check(value))
    {
        const char* value_str = PyUnicode_AsUTF8(value);
        if (strcmp(value_str, "exact") == 0)
            ((m00nny::TSNE*)self)->tsne_method = m00nny::TSNE::Exact;
        else if (strcmp(value_str, "barnes_hut") == 0)
            ((m00nny::TSNE*)self)->tsne_method = m00nny::TSNE::BarnesHut;
        else { PyErr_SetString(PyExc_ValueError, "Unknown tsne_method"); return -1; }
    }
    else { PyErr_SetString(PyExc_TypeError, "tsne_method must be a string"); return -1; }
    ExitGILScope
    return 0;
}