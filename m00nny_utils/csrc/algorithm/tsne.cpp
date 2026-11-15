#include "algorithm/tsne.h"

#include <ATen/CPUGeneratorImpl.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

namespace {

using m00nny::TSNE;

// Tuning (not part of the API).
constexpr int64_t kExaggerationIters       = 250;     // first phase: at most this, and at most a quarter of max_iter
constexpr double  kNeighboursPerPerplexity = 3.0;     // barnes_hut keeps 3 * perplexity + 1 neighbours per point
constexpr int64_t kPerplexitySteps         = 100;     // bisection steps per bandwidth
constexpr double  kPerplexityTol           = 1e-5;    // |entropy - log(perplexity)|
constexpr double  kInitScale               = 1e-4;    // standard deviation of the initial embedding
constexpr double  kSpectralShift           = 1e-6;    // mu relative to the mean diagonal of 4 a L(P)
constexpr double  kArmijo                  = 1e-4;    // sufficient decrease
constexpr int64_t kMaxBacktracks           = 30;      // halvings of the step
constexpr double  kMinGradNorm             = 1e-7;
constexpr double  kMinImprovement          = 1e-7;    // progress: a relative decrease of at least this
constexpr double  kCGTol                   = 1e-3;    // relative residual of the conjugate gradient solve
constexpr int64_t kCGMaxIters              = 100;
constexpr int64_t kDistanceBudget          = int64_t(1) << 25;   // elements of one batch of input distances
constexpr double  kTiny                    = std::numeric_limits<double>::min();

/*
Input side: P
*/

// Distances from rows [begin, end) of X to every row by the metric, the row's own point excluded (+inf);
// euclidean distances squared, as the Gaussian kernel of t-SNE takes them.
torch::Tensor
input_distances(torch::Tensor& X, int64_t begin, int64_t end, int8_t metric)
{
    torch::Tensor Xb = X.slice(0, begin, end);
    torch::Tensor D = m00nny::pairwise_distance(Xb, X, metric).to(torch::kDouble);
    if (metric == m00nny::Euclidean) D = D.square();
    torch::Tensor rows = torch::arange(end - begin, D.options().dtype(torch::kLong));
    D.index_put_({rows, rows + begin}, std::numeric_limits<double>::infinity());
    return D;
}

// p_{j|i} proportional to exp(-beta_i D_ij) over each row's candidates (D: N x K, +inf excluded), each beta_i by
// bisection so that the row's entropy is log(perplexity). Rows sum to 1.
torch::Tensor
conditional_probabilities(const torch::Tensor& D, double perplexity)
{
    const double target = std::log(perplexity);
    torch::Tensor D0  = D - D.amin(1, true);                                  // the nearest gets exp(0) = 1
    torch::Tensor D0f = torch::where(torch::isfinite(D0), D0, torch::zeros_like(D0));
    torch::Tensor beta = torch::ones({D.size(0), 1}, D.options());
    torch::Tensor lo = torch::zeros_like(beta);
    torch::Tensor hi = torch::full_like(beta, std::numeric_limits<double>::infinity());
    torch::Tensor P, S;
    for (int64_t step = 0; step < kPerplexitySteps; ++step)
    {
        P = torch::exp(-D0 * beta);
        S = P.sum(1, true);
        torch::Tensor excess = torch::log(S) + beta * (D0f * P).sum(1, true) / S - target;
        if (excess.abs().max().item<double>() < kPerplexityTol) break;
        torch::Tensor narrow = excess > 0;                                    // entropy too high: larger beta
        lo   = torch::where(narrow, beta, lo);
        hi   = torch::where(narrow, hi, beta);
        beta = torch::where(torch::isinf(hi), beta * 2, (lo + hi) / 2);
    }
    return P / S;
}

// Dense joint P = (P_cond + P_cond^T) / 2N (sums to 1), for exact.
torch::Tensor
joint_dense(torch::Tensor& X, double perplexity, int8_t metric, int64_t batch)
{
    const int64_t N = X.size(0);
    torch::Tensor Pc = torch::empty({N, N}, X.options().dtype(torch::kDouble));
    for (int64_t s = 0; s < N; s += batch)
    {
        const int64_t e = std::min(N, s + batch);
        Pc.slice(0, s, e).copy_(conditional_probabilities(input_distances(X, s, e, metric), perplexity));
    }
    return (Pc + Pc.t()) / (2.0 * N);
}

// Sparse joint P on the nearest neighbours, for barnes_hut: the edges (I, J, value) of the symmetrized matrix,
// each pair in both directions, and the matrix itself (CPU, float64).
struct Edges { torch::Tensor I, J, P, matrix; };

Edges
joint_sparse(torch::Tensor& X, double perplexity, int8_t metric, int64_t batch)
{
    const int64_t N = X.size(0);
    const int64_t k = std::min<int64_t>(N - 1, (int64_t)(kNeighboursPerPerplexity * perplexity) + 1);
    torch::Tensor dist = torch::empty({N, k}, X.options().dtype(torch::kDouble));
    torch::Tensor nbr  = torch::empty({N, k}, X.options().dtype(torch::kLong));
    for (int64_t s = 0; s < N; s += batch)
    {
        const int64_t e = std::min(N, s + batch);
        auto [v, j] = input_distances(X, s, e, metric).topk(k, 1, /*largest=*/false);
        dist.slice(0, s, e).copy_(v);
        nbr.slice(0, s, e).copy_(j);
    }
    torch::Tensor Pc   = conditional_probabilities(dist, perplexity).cpu().flatten();
    torch::Tensor rows = torch::arange(N, torch::kLong).repeat_interleave(k);
    torch::Tensor cols = nbr.cpu().flatten();
    torch::Tensor M = torch::sparse_coo_tensor(torch::stack({torch::cat({rows, cols}), torch::cat({cols, rows})}),
                                               torch::cat({Pc, Pc}) / (2.0 * N), {N, N}).coalesce();
    torch::Tensor idx = M.indices();
    return Edges{idx[0], idx[1], M.values(), M};
}

/*
Embedding side: the objective  f(Y) = sum aP log(aP) + sum aP log(1 + |y_i - y_j|^2) + a log Z,
Z = sum_{i != j} w_ij,  which is KL(P || Q) at a = 1, and its gradient
    grad_i = 4 sum_j (a p_ij - a w_ij / Z) w_ij (y_i - y_j).
*/
struct Evaluation { double f; torch::Tensor grad; };

Evaluation
exact_objective(const torch::Tensor& Y, const torch::Tensor& aP, double a, double plogp, bool with_grad)
{
    torch::Tensor sq = Y.square().sum(1);
    torch::Tensor D2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * Y.mm(Y.t())).clamp_min(0);
    torch::Tensor W  = (1 + D2).reciprocal();
    W.fill_diagonal_(0);
    const double Z = W.sum().item<double>();
    const double f = plogp + (aP * torch::log1p(D2)).sum().item<double>() + a * std::log(Z);
    if (!with_grad) return Evaluation{f, torch::Tensor()};
    torch::Tensor M = (aP - (a / Z) * W) * W;
    return Evaluation{f, 4 * (M.sum(1, true) * Y - M.mm(Y))};
}

// Attraction exact over the neighbour edges, repulsion and Z by the Barnes-Hut tree. Y: contiguous CPU float64.
Evaluation
barnes_hut_objective(const torch::Tensor& Y, const Edges& E, double a, double plogp, double angle, bool with_grad)
{
    const int64_t N = Y.size(0), d = Y.size(1);
    torch::Tensor diff = Y.index_select(0, E.I) - Y.index_select(0, E.J);
    torch::Tensor d2 = diff.square().sum(1);
    m00nny::BarnesHutTree tree(Y.data_ptr<double>(), N, d);
    torch::Tensor F  = torch::empty_like(Y);
    torch::Tensor Zi = torch::empty({N}, Y.options());
    tree.repulsion(angle, F.data_ptr<double>(), Zi.data_ptr<double>());
    const double Z = Zi.sum().item<double>();
    const double f = plogp + a * (E.P * torch::log1p(d2)).sum().item<double>() + a * std::log(Z);
    if (!with_grad) return Evaluation{f, torch::Tensor()};
    torch::Tensor attraction = torch::zeros_like(Y).index_add_(0, E.I, (a * E.P / (1 + d2)).unsqueeze(1) * diff);
    return Evaluation{f, 4 * (attraction - (a / Z) * F)};
}

/*
The second-order direction: B dY = -grad with B = 4 a L(P) + mu I (the same matrix for every column of Y).
*/

// exact: B dense, factored once per phase.
torch::Tensor
partial_hessian_factor(const torch::Tensor& P, double a)
{
    torch::Tensor B = 4 * a * (torch::diag(P.sum(1)) - P);
    B.diagonal().add_(kSpectralShift * B.diagonal().mean().item<double>() + kTiny);
    return torch::linalg_cholesky(B);
}

// barnes_hut: B applied through the sparse P, solved by conjugate gradients with the diagonal as preconditioner.
struct SparsePartialHessian
{
    torch::Tensor P, degree;
    double a, mu;
    torch::Tensor apply(const torch::Tensor& X) const
    { return 4 * this->a * (this->degree.unsqueeze(1) * X - torch::mm(this->P, X)) + this->mu * X; }
};

torch::Tensor
conjugate_gradient(const SparsePartialHessian& B, const torch::Tensor& b)
{
    torch::Tensor diag = (4 * B.a * B.degree + B.mu).unsqueeze(1);
    torch::Tensor x = torch::zeros_like(b), r = b.clone(), z = r / diag, p = z.clone();
    torch::Tensor rz = (r * z).sum(0);
    const double stop = kCGTol * b.norm().item<double>();
    for (int64_t it = 0; it < kCGMaxIters; ++it)
    {
        torch::Tensor Bp = B.apply(p);
        torch::Tensor alpha = rz / (p * Bp).sum(0).clamp_min(kTiny);
        x.add_(alpha * p);
        r.sub_(alpha * Bp);
        if (r.norm().item<double>() <= stop) break;
        z = r / diag;
        torch::Tensor rz_next = (r * z).sum(0);
        p = z + (rz_next / rz.clamp_min(kTiny)) * p;
        rz = rz_next;
    }
    return x;
}

} // namespace

/*
C++ class implementation
*/
torch::Tensor
m00nny::TSNE::fit_transform(const torch::Tensor& X_in)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::TSNE::fit_transform\n")
    torch::NoGradGuard no_grad;
    if (X_in.dim() != 2) throw std::invalid_argument("TSNE.fit_transform: X must be (n_samples, n_features)");
    const int64_t N = X_in.size(0), D = X_in.size(1), d = this->n_components;
    const bool barnes_hut = this->tsne_method == BarnesHut;
    if (N < 2) throw std::invalid_argument("TSNE.fit_transform: needs at least 2 samples");
    if (!(this->perplexity > 0) || this->perplexity >= N)
        throw std::invalid_argument("TSNE: perplexity must be in (0, n_samples)");
    if (d < 1) throw std::invalid_argument("TSNE: n_components must be >= 1");
    if (barnes_hut && d > 3) throw std::invalid_argument("TSNE: barnes_hut supports n_components <= 3; use tsne_method='exact'");
    if (!(this->learning_rate > 0) || !(this->early_exaggeration >= 1) || this->max_iter < 1 || this->batch_size < 1 || !(this->angle >= 0))
        throw std::invalid_argument("TSNE: learning_rate > 0, early_exaggeration >= 1, max_iter >= 1, batch_size >= 1, angle >= 0");

    torch::Tensor X = X_in.to(torch::kDouble);
    const torch::Device device = barnes_hut ? torch::Device(torch::kCPU) : X.device();
    const int64_t batch = std::max<int64_t>(1, std::min(this->batch_size, kDistanceBudget / std::max<int64_t>(1, N * D)));

    // P, and sum P log P (the constant of the objective).
    torch::Tensor P;
    Edges E;
    double plogp;
    if (barnes_hut) { E = joint_sparse(X, this->perplexity, this->dist_method, batch); plogp = torch::xlogy(E.P, E.P).sum().item<double>(); }
    else            { P = joint_dense(X, this->perplexity, this->dist_method, batch);  plogp = torch::xlogy(P, P).sum().item<double>(); }

    // Initial embedding, centred, standard deviation kInitScale.
    torch::Tensor Y;
    if (this->init_method == PCA)
    {
        if (std::min(N, D) < d) throw std::invalid_argument("TSNE: init_method='pca' needs n_samples and n_features >= n_components");
        torch::Tensor Xc = X - X.mean(0, true);
        auto [U, S, V] = m00nny::singular_value_decomposition(Xc, d);
        Y = Xc.mm(V);
        const double sd = Y.select(1, 0).std().item<double>();
        Y = Y * (kInitScale / std::max(sd, kTiny));
    }
    else if (this->init_method == Random)
    {
        at::Generator gen = at::make_generator<at::CPUGeneratorImpl>((uint64_t)this->random_state);
        Y = torch::randn({N, d}, gen, torch::TensorOptions().dtype(torch::kDouble)) * kInitScale;
    }
    else throw std::invalid_argument("TSNE: unknown init_method");
    Y = Y.to(device);
    Y = (Y - Y.mean(0, true)).contiguous();

    // Two phases: exaggerated P (a = early_exaggeration), then P (a = 1). Sum P = 1, so sum aP log(aP) = a plogp + a log a.
    struct Phase { double a; int64_t iters; bool last; };
    const int64_t exaggeration_iters = std::min(kExaggerationIters, this->max_iter / 4);
    const Phase phases[] = {{this->early_exaggeration, exaggeration_iters, false},
                            {1.0, this->max_iter - exaggeration_iters, true}};
    int64_t iterations = 0;
    double final_f = std::numeric_limits<double>::quiet_NaN();
    for (const Phase& phase : phases)
    {
        const double a = phase.a, aplogp = a * plogp + a * std::log(a);
        std::function<Evaluation(const torch::Tensor&, bool)> objective;
        std::function<torch::Tensor(const torch::Tensor&)> solve;
        if (barnes_hut)
        {
            torch::Tensor degree = torch::zeros({N}, E.P.options()).index_add_(0, E.I, E.P);
            const SparsePartialHessian B{E.matrix, degree, a, kSpectralShift * 4 * a * degree.mean().item<double>() + kTiny};
            const double angle = this->angle;
            objective = [&E, a, aplogp, angle](const torch::Tensor& Yv, bool g) { return barnes_hut_objective(Yv, E, a, aplogp, angle, g); };
            solve = [B](const torch::Tensor& b) { return conjugate_gradient(B, b); };
        }
        else
        {
            const torch::Tensor aP = a * P;
            const torch::Tensor factor = partial_hessian_factor(P, a);
            objective = [aP, a, aplogp](const torch::Tensor& Yv, bool g) { return exact_objective(Yv, aP, a, aplogp, g); };
            solve = [factor](const torch::Tensor& b) { return torch::cholesky_solve(b, factor); };
        }

        Evaluation now = objective(Y, true);
        double best = now.f;
        int64_t stalled = 0;
        for (int64_t i = 0; i < phase.iters; ++i)
        {
            torch::Tensor g = now.grad - now.grad.mean(0, true);         // translations leave f unchanged
            if (g.norm().item<double>() < kMinGradNorm) break;
            torch::Tensor step = solve(-g);
            step = step - step.mean(0, true);
            const double slope = (g * step).sum().item<double>();
            if (!(slope < 0)) break;                                     // no descent left at working precision

            double t = this->learning_rate;
            bool accepted = false;
            torch::Tensor trial;
            for (int64_t k = 0; k < kMaxBacktracks; ++k, t *= 0.5)
            {
                trial = (Y + t * step).contiguous();
                if (objective(trial, false).f <= now.f + kArmijo * t * slope) { accepted = true; break; }
            }
            if (!accepted) break;
            Y = trial;
            ++iterations;
            now = objective(Y, true);
            if (now.f < best - kMinImprovement * std::max(1.0, std::abs(best))) { best = now.f; stalled = 0; }
            else if (phase.last && ++stalled >= this->n_iter_without_progress) break;
        }
        if (phase.last) final_f = now.f;
    }
    this->kl_divergence = final_f;
    this->n_iter = iterations;
    return Y.to(X_in.options());
}

/*
Python C API Functions
*/

PyObject* 
TSNE_alloc
(PyTypeObject *type, Py_ssize_t nitems)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__alloc__\n")
    return PyType_GenericAlloc(type, nitems);
}

PyObject* 
TSNE_new
(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__new__\n")
    return PyType_GenericNew(type, args, kwds);   // zero-initialized; __init__ sets every field
}

int 
TSNE_init 
(m00nny::TSNE *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__init__\n")
    using m00nny::TSNE;
    using namespace m00nny::pyarg;
    const char* kwlist[] = {
        "n_components", "perplexity", "learning_rate", "early_exaggeration",
        "max_iter", "n_iter_without_progress", "batch_size", "init_method",
        "dist_method", "tsne_method", "angle", "random_state", NULL};
    PyObject *n_components = nullptr, *perplexity = nullptr, *learning_rate = nullptr, *early_exaggeration = nullptr,
             *max_iter = nullptr, *n_iter_without_progress = nullptr, *batch_size = nullptr, *init_method = nullptr,
             *dist_method = nullptr, *tsne_method = nullptr, *angle = nullptr, *random_state = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O$OOOOOOOOOOO", (char**)kwlist,
        &n_components, &perplexity, &learning_rate, &early_exaggeration,
        &max_iter, &n_iter_without_progress, &batch_size, &init_method,
        &dist_method, &tsne_method, &angle, &random_state)) return -1;

    if (!to_int   (n_components,            "n_components",            TSNE::kDefaultNComponents,          self->n_components)
     || !to_double(perplexity,              "perplexity",              TSNE::kDefaultPerplexity,           self->perplexity)
     || !to_double(learning_rate,           "learning_rate",           TSNE::kDefaultLearningRate,         self->learning_rate)
     || !to_double(early_exaggeration,      "early_exaggeration",      TSNE::kDefaultEarlyExaggeration,    self->early_exaggeration)
     || !to_int   (max_iter,                "max_iter",                TSNE::kDefaultMaxIter,              self->max_iter)
     || !to_int   (n_iter_without_progress, "n_iter_without_progress", TSNE::kDefaultNIterWithoutProgress, self->n_iter_without_progress)
     || !to_int   (batch_size,              "batch_size",              TSNE::kDefaultBatchSize,            self->batch_size)
     || !to_choice(init_method,             "init_method",             TSNE::kInitMethods,                 TSNE::kDefaultInitMethod, self->init_method)
     || !m00nny::metric_arg(dist_method,    "dist_method",             TSNE::kDefaultDistMethod,           self->dist_method)
     || !to_choice(tsne_method,             "tsne_method",             TSNE::kMethods,                     TSNE::kDefaultMethod,     self->tsne_method)
     || !to_double(angle,                   "angle",                   TSNE::kDefaultAngle,                self->angle)
     || !to_int   (random_state,            "random_state",            (int64_t)std::random_device{}(),    self->random_state))
        return -1;

    const char* invalid =
        self->n_components < 1               ? "n_components must be >= 1" :
        !(self->perplexity > 0)              ? "perplexity must be > 0" :
        !(self->learning_rate > 0)           ? "learning_rate must be > 0" :
        !(self->early_exaggeration >= 1)     ? "early_exaggeration must be >= 1" :
        self->max_iter < 1                   ? "max_iter must be >= 1" :
        self->n_iter_without_progress < 1    ? "n_iter_without_progress must be >= 1" :
        self->batch_size < 1                 ? "batch_size must be >= 1" :
        !(self->angle >= 0)                  ? "angle must be >= 0" : nullptr;
    if (invalid) { PyErr_SetString(PyExc_ValueError, invalid); return -1; }
    self->kl_divergence = std::numeric_limits<double>::quiet_NaN();
    self->n_iter = 0;
    return 0;
}

void
TSNE_free
(m00nny::TSNE *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__free__\n")
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
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__dealloc__\n")
    // Python allocates and frees the object's memory but never runs C++ destructors: run it here, so the
    // members' RAII releases what they own, then free the memory (tp_free: the default).
    self->~TSNE();
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject*
TSNE_call
(m00nny::TSNE *self, PyObject *args, PyObject *kwds)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__call__\n")
    return TSNE_fit_transform(self, args, kwds);
}

PyObject*
TSNE_str
(m00nny::TSNE *self)
{
    LIBM00NNY_DEBUG_MESSAGE("Python: TSNE.__str__\n")
    using m00nny::TSNE;
    LIBM00NNY_PY_TRY
    {
        std::ostringstream s;
        s << "<TSNE: n_components=" << self->n_components << ", perplexity=" << self->perplexity
          << ", learning_rate=" << self->learning_rate << ", early_exaggeration=" << self->early_exaggeration
          << ", max_iter=" << self->max_iter << ", n_iter_without_progress=" << self->n_iter_without_progress
          << ", batch_size=" << self->batch_size
          << ", init_method=" << m00nny::pyarg::name_of(TSNE::kInitMethods, self->init_method)
          << ", dist_method=" << m00nny::metric_str(self->dist_method)
          << ", tsne_method=" << m00nny::pyarg::name_of(TSNE::kMethods, self->tsne_method)
          << ", angle=" << self->angle << ", random_state=" << self->random_state << ">";
        return PyUnicode_FromString(s.str().c_str());
    }
    LIBM00NNY_PY_CATCH(NULL)
}

PyObject*
TSNE_repr
(m00nny::TSNE *self)
{
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
    const char* kwlist[] = {"X", NULL};
    PyObject* pyX = nullptr;
    torch::Tensor X;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)kwlist, &pyX)) return NULL;
    if (!m00nny::pyarg::to_tensor(pyX, "X", X)) return NULL;
    LIBM00NNY_PY_TRY { return THPVariable_Wrap(self->fit_transform(X)); }
    LIBM00NNY_PY_CATCH(NULL)
}

/*
Getters and setters of the named settings (the names: TSNE::kInitMethods, TSNE::kMethods, m00nny::kMetricNames)
*/
template <std::size_t N>
static int
set_choice(PyObject* value, const char* name, const std::pair<const char*, int8_t> (&names)[N], int8_t& field)
{
    if (!m00nny::pyarg::given(value)) { PyErr_Format(PyExc_TypeError, "%s must be a str", name); return -1; }
    return m00nny::pyarg::to_choice(value, name, names, field, field) ? 0 : -1;
}

PyObject*
TSNE_get_init_method
(PyObject *self, void* closure)
{
    return PyUnicode_FromString(m00nny::pyarg::name_of(m00nny::TSNE::kInitMethods, ((m00nny::TSNE*)self)->init_method));
}

PyObject*
TSNE_get_dist_method
(PyObject *self, void* closure)
{
    LIBM00NNY_PY_TRY { return PyUnicode_FromString(m00nny::metric_str(((m00nny::TSNE*)self)->dist_method)); }
    LIBM00NNY_PY_CATCH(NULL)
}

PyObject*
TSNE_get_tsne_method
(PyObject *self, void* closure)
{
    return PyUnicode_FromString(m00nny::pyarg::name_of(m00nny::TSNE::kMethods, ((m00nny::TSNE*)self)->tsne_method));
}

int
TSNE_set_init_method
(PyObject *self, PyObject *value, void* closure)
{
    return set_choice(value, "init_method", m00nny::TSNE::kInitMethods, ((m00nny::TSNE*)self)->init_method);
}

int
TSNE_set_dist_method
(PyObject *self, PyObject *value, void* closure)
{
    if (!m00nny::pyarg::given(value)) { PyErr_SetString(PyExc_TypeError, "dist_method must be a metric name or a non-negative int"); return -1; }
    int8_t& field = ((m00nny::TSNE*)self)->dist_method;
    return m00nny::metric_arg(value, "dist_method", field, field) ? 0 : -1;
}

int
TSNE_set_tsne_method
(PyObject *self, PyObject *value, void* closure)
{
    return set_choice(value, "tsne_method", m00nny::TSNE::kMethods, ((m00nny::TSNE*)self)->tsne_method);
}
