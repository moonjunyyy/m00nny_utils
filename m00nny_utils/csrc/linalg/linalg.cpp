#include "linalg/linalg.h"

#include <ATen/CPUGeneratorImpl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

// Tuning of the eigen solver (not part of the API).
constexpr double   kSymmetryTolerance = 1e3;                 // allowed asymmetry, in units of the input's roundoff
constexpr uint64_t kEigenStartSeed    = 0x6d30306e6e79ULL;   // fixed random start: the result is a function of X alone
constexpr double   kSignTieTol        = 1e-6;                // entries this close (relative) to the largest count as tied
constexpr double   kTrustRadius0      = 0.5;                 // first trust radius, times sqrt(r) (V has unit columns)
constexpr double   kTrustRadiusMax    = 2.0;                 // largest trust radius, times sqrt(r)
constexpr double   kTrustRadiusMin    = 1e-14;               // a matrix whose radius falls below this has stopped moving
constexpr double   kRoundingUnits     = 100.0;               // loss changes below this many roundoffs of |X|_F^2 are not resolved

double
unit_roundoff(torch::ScalarType t)
{
    switch (t)
    {
        case torch::kDouble:   return std::numeric_limits<double>::epsilon();
        case torch::kHalf:     return 9.765625e-4;   // 2^-10
        case torch::kBFloat16: return 7.8125e-3;     // 2^-7
        default:               return std::numeric_limits<float>::epsilon();
    }
}

std::vector<int64_t>
with_batch(const torch::Tensor& X, std::initializer_list<int64_t> tail)
{
    std::vector<int64_t> shape(X.sizes().begin(), X.sizes().end() - 2);
    shape.insert(shape.end(), tail);
    return shape;
}

torch::Tensor sym(const torch::Tensor& M) { return 0.5 * (M + M.transpose(1, 2)); }

// Onto the tangent space at V of the matrices with orthonormal columns (the directions orthogonalize keeps).
torch::Tensor tangent(const torch::Tensor& V, const torch::Tensor& Z)
{ return Z - torch::bmm(V, sym(torch::bmm(V.transpose(1, 2), Z))); }

// Inner product per matrix, (B, n, r) x (B, n, r) -> (B); and a (B) value broadcast over (B, n, r).
torch::Tensor inner_product(const torch::Tensor& X, const torch::Tensor& Y) { return (X * Y).sum({1, 2}); }
torch::Tensor per_matrix(const torch::Tensor& x) { return x.view({-1, 1, 1}); }

} // namespace

void
m00nny::internal::check_square(const torch::Tensor& X, const std::string& func_name) {
    if (X.dim() < 2) {
        throw std::invalid_argument(func_name + ": Input tensor must have at least 2 dimensions. Got " + std::to_string(X.dim()));
    }
    int64_t n = X.size(-2);
    int64_t m = X.size(-1);
    if (n != m) {
        throw std::invalid_argument(func_name + ": Input tensor's last two dimensions must be square. Got (" +
                                    std::to_string(n) + ", " + std::to_string(m) + ")");
    }
}

void
m00nny::internal::check_symmetric(const torch::Tensor& A, double roundoff, const std::string& func_name)
{
    torch::Tensor scale = A.abs().amax({1, 2});
    torch::Tensor asym  = (A - A.transpose(1, 2)).abs().amax({1, 2});
    if ((asym > kSymmetryTolerance * roundoff * scale).any().item<bool>())
        throw std::invalid_argument(func_name + ": the matrix must be symmetric (max |X - X^T| = " +
                                    std::to_string(asym.max().item<double>()) + ")");
}

torch::Tensor
m00nny::inverse(const torch::Tensor& X_const) {
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::inverse\n");
    m00nny::internal::check_square(X_const, "m00nny::inverse");

    int64_t B = 1;
    auto dims = X_const.dim();
    const int64_t N = X_const.size(-1);
    for (int64_t i = 0; i < dims - 2; ++i) { B *= X_const.size(i); }

    const double epsilon = 1e-12;
    auto options = X_const.options();
    
    // Initial augmented matrix (B, N, 2N)
    torch::Tensor A_aug = torch::cat({
        X_const.clone().reshape({B, N, N}),
        torch::eye(N, options).unsqueeze(0).expand({B, N, N}).clone()
    }, 2);

    // Row permutation buffer (B, N)
    torch::Tensor row_perm = torch::arange(N, options.dtype(torch::kLong)).unsqueeze(0).expand({B, N}).clone();
    torch::Tensor batch_idx = torch::arange(B, options.dtype(torch::kLong));
    torch::Tensor all_k = torch::arange(N, options.dtype(torch::kLong));
    torch::Tensor mask = torch::empty({B, N}, options.dtype(torch::kBool));
    
    // Buffers
    torch::Tensor long_buffer1 = torch::empty({B, N}, options.dtype(torch::kLong)); // Not used in this version, but kept if needed

    // Re-sized buffers to be more specific if N is small. Max N rows needed.
    torch::Tensor float_buffer1 = torch::empty({B, N, 2 * N}, options); // Used for slice_rows, pivot_row_data, rows_to_update
    torch::Tensor float_buffer2 = torch::empty({B, N, 2 * N}, options); // Used for col_i intermediate
    torch::Tensor float_buffer3 = torch::empty({B, N, 2 * N}, options); // Used for pivot_row_normalized

    for (int64_t i = 0; i < N; ++i) {
        // === 1. Find pivot ===
        torch::Tensor remaining_rows_perm_indices = row_perm.slice(1, i, N); // (B, N-i). Original row indices at permuted positions i..N-1.
        
        // Use a slice of float_buffer1 for slice_rows
        torch::Tensor slice_rows = float_buffer1.slice(1, 0, N - i); // (B, N-i, 2N)
        torch::gather_out(slice_rows, A_aug, 1, remaining_rows_perm_indices.unsqueeze(-1).expand({B, N - i, 2 * N}));
        
        // Use a slice of float_buffer2 for col_i magnitude storage
        torch::Tensor col_i_magnitudes = float_buffer2.slice(1, 0, N-i).slice(2,0,1).select(2,0); // (B, N-i)
        col_i_magnitudes.copy_(slice_rows.select(2, i).abs()); // Get column 'i' of the selected rows
        
        auto [_, rel_pivot_pos] = col_i_magnitudes.max(1); // (B,) indices relative to the (N-i) slice
        torch::Tensor abs_pivot_perm_idx = rel_pivot_pos + i;  // (B,) permuted index of the chosen pivot row

        // === 2. row_perm swap ===
        // Swap original row indices in `row_perm` at permuted positions `i` and `abs_pivot_perm_idx`
        torch::Tensor current_val_at_perm_i = row_perm.index({batch_idx, torch::full({B}, i, row_perm.options().dtype(torch::kLong))}).clone(); // (B,)
        torch::Tensor pivot_val_at_perm_pivot_idx = row_perm.index({batch_idx, abs_pivot_perm_idx}); // (B,)
        row_perm.index_put_({batch_idx, i}, pivot_val_at_perm_pivot_idx);
        row_perm.index_put_({batch_idx, abs_pivot_perm_idx}, current_val_at_perm_i);

        // === 3. Normalize pivot row ===
        // Get the *original row index* of the row that is now at permuted position `i`. This is the true pivot row index in A_aug.
        torch::Tensor pivot_row_original_indices = row_perm.index({batch_idx, i}); // Shape (B,)

        // For gather/scatter, indices often need to be (B,1). Let's make a (B,1) version.
        torch::Tensor pivot_row_original_indices_B1 = pivot_row_original_indices.unsqueeze(1); // Shape (B,1)
        // Original line that had Error 0:
        // torch::Tensor pivot_row_indices = long_buffer2.slice(1, 0, 1);
        // pivot_row_indices.copy_(row_perm.index({batch_idx, i})); // ERROR 0: Shape mismatch (B,1) vs (B,)
        // Corrected implicit fix: We now use pivot_row_original_indices_B1 which is correctly shaped.
        
        // Gather the actual pivot row data from A_aug into a temporary buffer (a slice of float_buffer1).
        torch::Tensor current_pivot_row_data = float_buffer1.slice(1,0,1); // (B,1,2N)
        current_pivot_row_data.copy_(A_aug.gather(1, pivot_row_original_indices_B1.unsqueeze(2).expand({B, 1, 2*N})));
        // Original: torch::Tensor pivot_rows = A_aug.gather(1, pivot_row_indices.unsqueeze(-1).expand({B, 1, 2*N})); (This line was fine if pivot_row_indices was (B,1))

        // Extract the pivot element value: A_aug[true_pivot_idx, i]
        // current_pivot_row_data is (B,1,2N). We need element (b,0,i). Result (B,).
        torch::Tensor pivot_elements = current_pivot_row_data.select(2,i).squeeze(1);
        // Original: torch::Tensor pivot_vals = pivot_rows.index({torch::indexing::Slice(), i}); // INDEXING ERROR 1
        
        if (pivot_elements.abs().lt(epsilon).any().item<bool>()) {
            throw std::runtime_error("Singular matrix encountered.");
        }

        // Normalize the pivot row. Store in a slice of float_buffer3.
        torch::Tensor normalized_pivot_row = float_buffer3.slice(1, 0, 1); // (B,1,2N)
        // Divisor `pivot_elements` is (B,). Needs to be (B,1,1) for broadcasting with (B,1,2N) dividend and quotient.
        torch::div_out(normalized_pivot_row, current_pivot_row_data, pivot_elements.unsqueeze(1).unsqueeze(2));
        // Original: torch::div_out(pivot_rows_normalized, pivot_rows, pivot_vals.unsqueeze(-1).expand({B, 1, 2*N})); // INDEXING ERROR 2 (divisor shape)
        
        // Scatter the normalized pivot row back into A_aug at its true original index.
        // Index for scatter: pivot_row_original_indices_B1 (B,1). Needs to be (B,1,2N) to match src shape.
        A_aug.scatter_(1, pivot_row_original_indices_B1.unsqueeze(2).expand_as(normalized_pivot_row), normalized_pivot_row);
        // Original: A_aug.scatter_(1, pivot_row_indices, pivot_rows_normalized.unsqueeze(1)); // INDEXING ERROR 3 (src shape for scatter)
        
        // === 4. Remove pivot column from all other rows (Gauss-Jordan elimination step) ===
        // This part was incomplete and had potential buffer aliasing in the original code.
        
        // Create a mask for permuted row indices `k` where `k != i`.
        torch::not_equal_out(mask, all_k.unsqueeze(0).expand_as(mask), i); // mask is (B,N), true where k_perm != i
        
        // Get permuted indices `k` that are not `i`.
        torch::Tensor other_permuted_indices_k = all_k.masked_select(mask[0]); // (N-1) tensor

        if (N - 1 > 0) { // Only proceed if there are other rows to update
            // Get original row indices for these "other" rows.
            torch::Tensor other_rows_original_indices = row_perm.index({ // Shape (B, N-1)
                batch_idx.unsqueeze(1), 
                other_permuted_indices_k.unsqueeze(0).expand({B, N-1})
            });

            // Prepare index for gathering/scattering these N-1 rows from/to A_aug.
            torch::Tensor gather_scatter_idx_for_other_rows = other_rows_original_indices.unsqueeze(-1).expand({B, N-1, 2*N}); // (B, N-1, 2N)
            
            // Use a slice of float_buffer1 for rows_to_update.
            // This slice might overlap with `current_pivot_row_data` if N-1 >= 1.
            // However, `current_pivot_row_data` (from float_buffer1) was used to compute `normalized_pivot_row` (in float_buffer3).
            // The crucial data for this step is `normalized_pivot_row`, which is in a different buffer. So, reusing float_buffer1 here is safe.
            torch::Tensor rows_to_update = float_buffer1.slice(1, 0, N-1); // (B, N-1, 2N)
            rows_to_update.copy_(A_aug.gather(1, gather_scatter_idx_for_other_rows)); // Fetch current state of these rows.

            // Calculate scaling factors: For each row_to_update, its element in column `i`.
            torch::Tensor factors = rows_to_update.select(2,i).clone(); // (B, N-1). Clone to be safe if select is a view for in-place ops.

            // Subtract factor * normalized_pivot_row from each row_to_update.
            // factors.unsqueeze(2) is (B,N-1,1)
            // normalized_pivot_row is (B,1,2N)
            // Product by broadcasting: (B,N-1,1) * (B,1,2N) -> (B,N-1,2N)
            rows_to_update.sub_(factors.unsqueeze(2) * normalized_pivot_row);

            // Scatter updated rows back to A_aug.
            A_aug.scatter_(1, gather_scatter_idx_for_other_rows, rows_to_update);
        }
    } // End of loop over i

    // === 5. Reorder output rows ===
    // The inverse part of A_aug is now computed, but rows are permuted according to `row_perm`.
    // We need to apply `row_perm` to get the final order.
    // `row_perm[b, j] = original_idx` means final row `j` should be `A_aug[b, original_idx, :]`.
    // This is a bit tricky. `row_perm` stores `original_index = row_perm[batch, final_row_index]`.
    // We need to create an inverse permutation or use scatter.
    // A simpler way for final reordering: if row_perm[b, k] = orig_idx, this means original row `orig_idx` is now at permuted row `k`.
    // So, the row `A_aug[b, orig_idx, :]` should end up in `Output[b, k, :]`.
    // This implies we need to sort `A_aug` based on `row_perm`.
    // `final_indices` should map `final_row_pos -> current_row_pos_in_A_aug (which is original_idx)`
    // This is exactly what `row_perm` provides.
    // Example: row_perm[b] = [1,0,2] (for N=3). Means:
    //   - Final row 0 content is from A_aug's original row 1.
    //   - Final row 1 content is from A_aug's original row 0.
    //   - Final row 2 content is from A_aug's original row 2.
    // So, gather A_aug rows using row_perm as indices.
    torch::Tensor final_indices = row_perm.unsqueeze(-1).expand({B, N, 2*N});
    torch::Tensor reordered_A_aug = A_aug.gather(1, final_indices);
    
    return reordered_A_aug.slice(2, N, 2*N).reshape_as(X_const);
}

torch::Tensor
m00nny::orthogonalize(const torch::Tensor& X_const)
{
    /*
    Make the input tensor orthogonal.
    This function assumes the input tensor is a matrix with columns as vectors.
    This function uses the Gram-Schmidt Process.

    Args:
        X (torch::Tensor): The input tensor. (*, n, m)

    Returns:
        torch::Tensor: The output tensor. (*, n, m)
    */

    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::orthogonalize\n")
    c10::IntArrayRef size = X_const.sizes();
    if (X_const.dim() < 2) {
        throw std::runtime_error("m00nny::orthogonalize: Input tensor must have at least 2 dimensions.");
    }
    int64_t n_rows = size[X_const.dim() - 2]; // Dimension of each vector
    int64_t m_cols = size[X_const.dim() - 1]; // Number of vectors

    if (m_cols == 0) return X_const.clone();

    int64_t num_matrix = 1;
    for (int64_t i = 0; i < X_const.dim() - 2; i++) num_matrix *= size[i];
    
    torch::Tensor A = X_const.reshape({num_matrix, n_rows, m_cols}).clone(); 
    const double epsilon = 1e-12;

    // Step 1: Normalize all input columns (vectors) individually. A_j = v_j / ||v_j||
    torch::Tensor col_norms_A = A.norm(2, /*dim=*/1, /*keepdim=*/true); // Norm along rows_dim for each column vector
    A.div_(col_norms_A.clamp_min(epsilon)); // A now contains a_1, a_2, ...

    // Step 2: C_j accumulates what is removed from column j: the cumulative sum, over the bases q_k fixed before j,
    //         of each basis' subtraction term <q_k, A_j - C_j> q_k. The terms are what is summed, not the vectors
    //         a_k: the projection onto a sum of directions is not the sum of the projections onto them.
    torch::Tensor C = torch::zeros_like(A);
    for (int64_t k = 0; k < m_cols; ++k)
    {
        // Step 3: Fix basis k: column k with its accumulated subtraction removed, normalized.
        torch::Tensor q = A.select(/*dim=*/2, k) - C.select(/*dim=*/2, k);
        q.div_(q.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(epsilon));
        A.select(/*dim=*/2, k).copy_(q);
        // Step 4: Add its subtraction term to every later column at once (they are computed in the space without
        //         bases 1..k from here on).
        if (k + 1 < m_cols)
        {
            torch::Tensor rest = A.slice(/*dim=*/2, k + 1) - C.slice(/*dim=*/2, k + 1);
            C.slice(/*dim=*/2, k + 1).add_(q.unsqueeze(2) * (q.unsqueeze(2) * rest).sum(/*dim=*/1, /*keepdim=*/true));
        }
    }
    return A.reshape(size);
}

namespace {

// The second-order update of eigen_decomposition (see there), batched, from the start V (B, n, r) with orthonormal
// columns, on the symmetric float64 batch A (B, n, n). Returns V.
torch::Tensor
newton_eigenvectors(const torch::Tensor& A, torch::Tensor V, int64_t max_iters, double threshold)
{
    using m00nny::orthogonalize;
    const int64_t B = A.size(0), n = A.size(1), r = V.size(2);
    const double tiny = std::numeric_limits<double>::min();
    const torch::Tensor normA2   = A.square().sum({1, 2});                          // |X|_F^2 per matrix, (B)
    const torch::Tensor tol      = (threshold * normA2.sqrt()).unsqueeze(1);        // (B, 1)
    const torch::Tensor rounding = kRoundingUnits * unit_roundoff(torch::kDouble) * normA2;
    const double radius_max = kTrustRadiusMax * std::sqrt((double)r);
    const int64_t max_cg = n * r;                                                   // the dimension bounds CG

    torch::Tensor radius = torch::full({B}, kTrustRadius0 * std::sqrt((double)r), A.options());
    torch::Tensor running = torch::ones({B}, A.options().dtype(torch::kBool));

    for (int64_t it = 0; it < max_iters; ++it)
    {
        const torch::Tensor AV  = torch::bmm(A, V);
        const torch::Tensor lam = (V * AV).sum(1);                                 // (B, r) Rayleigh quotients
        const torch::Tensor res = (AV - V * lam.unsqueeze(1)).norm(2, 1);          // (B, r)
        running = running & (res > tol).any(1) & (radius >= kTrustRadiusMin);
        if (!running.any().item<bool>()) break;

        const torch::Tensor f    = normA2 - lam.square().sum(1);                   // L1 on orthonormal V
        const torch::Tensor G    = -4 * AV * lam.unsqueeze(1);                     // Euclidean gradient
        const torch::Tensor S    = sym(torch::bmm(V.transpose(1, 2), G));
        const torch::Tensor grad = G - torch::bmm(V, S);                           // = tangent(V, G)
        auto hess = [&](const torch::Tensor& W)
        {
            const torch::Tensor dlam = 2 * (W * AV).sum(1);
            return tangent(V, -4 * torch::bmm(A, W) * lam.unsqueeze(1) - 4 * AV * dlam.unsqueeze(1) - torch::bmm(W, S));
        };

        // Steihaug-Toint conjugate gradients on  <grad, W> + 1/2 <W, Hess W>,  |W| <= radius,  per matrix.
        const torch::Tensor gnorm = inner_product(grad, grad).sqrt();
        const torch::Tensor eta   = (gnorm / normA2.clamp_min(tiny)).sqrt().clamp_max(0.5);   // forcing term
        torch::Tensor W = torch::zeros_like(V), R = -grad, P = R.clone(), rr = gnorm.square();
        torch::Tensor solving = running.clone();
        auto to_boundary = [&](const torch::Tensor& W0, const torch::Tensor& P0)
        {
            const torch::Tensor a = inner_product(P0, P0), b = 2 * inner_product(W0, P0), c = inner_product(W0, W0) - radius.square();
            return W0 + per_matrix((-b + (b.square() - 4 * a * c).clamp_min(0).sqrt()) / (2 * a).clamp_min(tiny)) * P0;
        };
        for (int64_t j = 0; j < max_cg && solving.any().item<bool>(); ++j)
        {
            const torch::Tensor HP    = hess(P);
            const torch::Tensor pHp   = inner_product(P, HP);
            const torch::Tensor alpha = rr / pHp.abs().clamp_min(tiny);
            const torch::Tensor Wn    = W + per_matrix(alpha) * P;
            const torch::Tensor hit    = solving & ((pHp <= 0) | (inner_product(Wn, Wn).sqrt() >= radius));
            const torch::Tensor inside = solving & ~hit;
            W = torch::where(per_matrix(hit), to_boundary(W, P), torch::where(per_matrix(inside), Wn, W));
            const torch::Tensor Rn  = R - per_matrix(alpha) * HP;
            const torch::Tensor rrn = inner_product(Rn, Rn);
            const torch::Tensor done = inside & (rrn.sqrt() <= eta * gnorm);
            R  = torch::where(per_matrix(inside), Rn, R);
            P  = torch::where(per_matrix(inside & ~done), Rn + per_matrix(rrn / rr.clamp_min(tiny)) * P, P);
            rr = torch::where(inside, rrn, rr);
            solving = inside & ~done;
        }

        // Take the step where the loss fell by enough of what the model predicted; adapt the radius.
        const torch::Tensor predicted = -(inner_product(grad, W) + 0.5 * inner_product(W, hess(W)));
        const torch::Tensor Vn  = orthogonalize(V + W);
        const torch::Tensor fn  = normA2 - (Vn * torch::bmm(A, Vn)).sum(1).square().sum(1);
        const torch::Tensor rho = torch::where(predicted <= rounding,
                                               torch::where(fn <= f + rounding, torch::ones_like(f), -torch::ones_like(f)),
                                               (f - fn) / predicted.clamp_min(tiny));
        const torch::Tensor step = inner_product(W, W).sqrt();
        radius = torch::where(running & (rho < 0.25), 0.25 * radius,
                 torch::where(running & (rho > 0.75) & (step >= 0.99 * radius), (2 * radius).clamp_max(radius_max), radius));
        V = torch::where(per_matrix(running & (rho > 0.1)), Vn, V);
    }
    return V;
}

// Eigenvalues in descending order with their eigenvectors; each eigenvector's sign fixed.
std::tuple<torch::Tensor, torch::Tensor>
sorted_eigenpairs(const torch::Tensor& A, torch::Tensor V)
{
    const int64_t B = A.size(0), n = A.size(1), r = V.size(2);
    const torch::Tensor lam = (V * torch::bmm(A, V)).sum(1);
    auto [lam_sorted, perm] = lam.sort(/*dim=*/1, /*descending=*/true);
    V = V.gather(2, perm.unsqueeze(1).expand({B, n, r}));
    // Sign: the first entry of (nearly) largest magnitude is positive. Ties within kSignTieTol go to the first, so
    // rounding cannot flip an eigenvector whose largest entries are equal in magnitude.
    torch::Tensor magnitude = V.abs();
    torch::Tensor leading = (magnitude >= magnitude.amax(1, true) * (1 - kSignTieTol)).to(torch::kInt8).argmax(1, true);
    torch::Tensor sign = V.gather(1, leading).sign();                   // (B, 1, r)
    V = V * torch::where(sign.eq(0), torch::ones_like(sign), sign);
    return std::make_tuple(lam_sorted, V);
}

// r eigenpairs of the symmetric float64 batch A (B, n, n), from the fixed-seed random start. square_first: select the
// subspace on A^2 first (eigenvalues l^2, the same eigenvectors), then continue on A from there. The loss on A has
// local minima when A is indefinite and r < n: a column whose Rayleigh quotient is negative reaches a larger |l| of
// the other sign only through a quotient of 0, where the loss is higher; on A^2 there is no sign, and from the
// right subspace the update on A stays in it.
std::tuple<torch::Tensor, torch::Tensor>
eigenpairs(const torch::Tensor& A, int64_t r, int64_t max_iters, double threshold, bool square_first)
{
    const int64_t B = A.size(0), n = A.size(1);
    at::Generator gen = at::make_generator<at::CPUGeneratorImpl>(kEigenStartSeed);
    torch::Tensor V = m00nny::orthogonalize(torch::randn({B, n, r}, gen, torch::TensorOptions().dtype(torch::kDouble)).to(A.options()));
    if (square_first)
    {
        const torch::Tensor A2 = torch::bmm(A, A);
        V = newton_eigenvectors(0.5 * (A2 + A2.transpose(1, 2)), V, max_iters, threshold);
    }
    V = newton_eigenvectors(A, V, max_iters, threshold);
    return sorted_eigenpairs(A, V);
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor>
m00nny::eigen_decomposition(const torch::Tensor& X, int64_t lowrank, int64_t max_iters, double threshold)
{
    /*
    Eigen decomposition of a real symmetric matrix X (*, n, n):  X V = V L,  V^T V = I.

    Loss (the residual of the decomposition):  L1 = |X - V L V^T|_F^2, with V kept orthonormal (orthogonalize after
    every step, which is what L2 = |V^T V - I|^2 asks for) and L = diag(V^T X V), the Rayleigh quotients, which
    minimize L1 for the given V. On orthonormal V it reads  L1 = |X|_F^2 - sum_i l_i^2 : its minimum is reached by
    the r eigenpairs of largest |l| (with all n pairs, by any eigenbasis).

    Second-order update: Newton on L1 over orthonormal V, i.e. with the gradient and the Hessian taken along the
    directions orthogonalize keeps (projected by P(Z) = Z - V sym(V^T Z)):
        G = -4 X V diag(l),     grad = G - V sym(V^T G),
        Hess[W] = P( -4 X W diag(l) - 4 X V diag(2 diag(W^T X V)) - W sym(V^T G) ).
    The Newton system is solved by conjugate gradients inside a trust region (Steihaug-Toint): a direction of negative
    curvature, as near a saddle, is followed to the region's boundary, and the radius follows how well the quadratic
    model predicted the decrease. The step is retracted by orthogonalize.

    lowrank r < n: the loss on X has local minima when X is indefinite, holding a pair of smaller |l| and opposite sign
    in place of a larger one (between the two the Rayleigh quotient passes through 0, where the loss is higher). So the
    subspace is selected on X^2 first (eigenvalues l^2, the same eigenvectors, no sign), and the update continues on X
    from there; each stage is bounded by max_iters.

    Computed in float64; returned in X's dtype, on X's device. The random start comes from a fixed seed, so the
    result depends on X alone; an eigenvector's sign is fixed so that its first entry of largest magnitude is positive.

    Args:
        X (torch::Tensor): (*, n, n), symmetric.
        lowrank (int64_t): number of eigenpairs r; <= 0 or > n means all n.
        max_iters (int64_t): bound on the Newton iterations.
        threshold (double): a pair is converged when |X v - l v| <= threshold * |X|_F.

    Returns:
        std::tuple<torch::Tensor, torch::Tensor>: L (*, r, r) diagonal, eigenvalues in descending order;
                                                  V (*, n, r), the eigenvectors as columns.
    */
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::eigen_decomposition\n")
    torch::NoGradGuard no_grad;
    internal::check_square(X, "m00nny::eigen_decomposition");
    const int64_t n = X.size(-1);
    const int64_t r = (lowrank <= 0 || lowrank > n) ? n : lowrank;

    torch::Tensor A = X.reshape({-1, n, n}).to(torch::kDouble);
    const int64_t B = A.size(0);
    if (B == 0 || r == 0)
        return std::make_tuple(torch::zeros(with_batch(X, {r, r}), X.options()),
                               torch::zeros(with_batch(X, {n, r}), X.options()));
    internal::check_symmetric(A, unit_roundoff(X.scalar_type()), "m00nny::eigen_decomposition");
    A = 0.5 * (A + A.transpose(1, 2));

    auto [lam, V] = eigenpairs(A, r, max_iters, threshold, /*square_first=*/r < n);

    return std::make_tuple(torch::diag_embed(lam).to(X.scalar_type()).reshape(with_batch(X, {r, r})),
                           V.to(X.scalar_type()).reshape(with_batch(X, {n, r})));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
m00nny::singular_value_decomposition(const torch::Tensor& X, int64_t lowrank, int64_t max_iters, double threshold)
{
    /*
    Thin singular value decomposition  X = U S V^T  of X (*, n, m), from the eigen decomposition of the smaller
    Gram matrix:  m <= n:  X^T X = V S^2 V^T,  U = X V S^-1;   m > n:  X X^T = U S^2 U^T,  V = X^T U S^-1,
    the derived factor orthonormalized again by Gram-Schmidt.
    k = min(lowrank, n, m) pairs (lowrank <= 0: all), singular values in descending order. A singular value at
    the Gram matrix's resolution (below max(n, m) * roundoff * s_1) gives a zero column in the derived factor.
    The Gram matrix squares the condition number: relative to s_1, singular values below about the square root of
    the float64 roundoff are not resolved.

    Returns:
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>: U (*, n, k), S (*, k, k) diagonal, V (*, m, k).
    */
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::singular_value_decomposition\n")
    torch::NoGradGuard no_grad;
    if (X.dim() < 2) throw std::invalid_argument("m00nny::singular_value_decomposition: Input tensor must have at least 2 dimensions.");
    const int64_t n = X.size(-2), m = X.size(-1);
    const int64_t kmax = std::min(n, m);
    const int64_t k = (lowrank <= 0 || lowrank > kmax) ? kmax : lowrank;
    const bool tall = m <= n;

    torch::Tensor A = X.reshape({-1, n, m}).to(torch::kDouble);
    torch::Tensor G = tall ? torch::bmm(A.transpose(1, 2), A) : torch::bmm(A, A.transpose(1, 2));
    if (k == 0 || A.size(0) == 0)
        return std::make_tuple(torch::zeros(with_batch(X, {n, k}), X.options()), torch::zeros(with_batch(X, {k, k}), X.options()),
                               torch::zeros(with_batch(X, {m, k}), X.options()));
    // The Gram matrix is positive semidefinite: no sign barrier, so no squaring (it would take s to the 4th power).
    auto [L, W] = eigenpairs(0.5 * (G + G.transpose(1, 2)), k, max_iters, threshold, /*square_first=*/false);
    torch::Tensor s = L.clamp_min(0).sqrt();                                      // (B, k); W: (B, m or n, k)
    torch::Tensor other = tall ? torch::bmm(A, W) : torch::bmm(A.transpose(1, 2), W);   // U S or V S
    torch::Tensor floor_s = (double)std::max(n, m) * unit_roundoff(torch::kDouble) * s.narrow(1, 0, std::min<int64_t>(1, k));
    torch::Tensor resolved = (s > floor_s).unsqueeze(1);
    other = torch::where(resolved, other / s.clamp_min(std::numeric_limits<double>::min()).unsqueeze(1), torch::zeros_like(other));
    // The derived factor is orthonormal only as far as the eigenvectors are converged, amplified by s_1 / s_k:
    // orthonormalize it at working precision (unresolved columns stay zero).
    other = orthogonalize(other);

    torch::Tensor U = tall ? other : W, V = tall ? W : other;
    return std::make_tuple(U.to(X.scalar_type()).reshape(with_batch(X, {n, k})),
                           torch::diag_embed(s).to(X.scalar_type()).reshape(with_batch(X, {k, k})),
                           V.to(X.scalar_type()).reshape(with_batch(X, {m, k})));
}

/*
Python bindings (argument reading and error translation: common/pyarg.h)
*/

PyObject*
linalg_inverse
(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", NULL};
    PyObject* pyX = nullptr;
    torch::Tensor X;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)keywords, &pyX)) return NULL;
    if (!m00nny::pyarg::to_tensor(pyX, "X", X)) return NULL;
    LIBM00NNY_PY_TRY { return THPVariable_Wrap(m00nny::inverse(X)); }
    LIBM00NNY_PY_CATCH(NULL)
}

PyObject*
linalg_orthogonalize
(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", NULL};
    PyObject* pyX = nullptr;
    torch::Tensor X;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)keywords, &pyX)) return NULL;
    if (!m00nny::pyarg::to_tensor(pyX, "X", X)) return NULL;
    LIBM00NNY_PY_TRY { return THPVariable_Wrap(m00nny::orthogonalize(X)); }
    LIBM00NNY_PY_CATCH(NULL)
}

// lowrank, max_iters, threshold: the optional, keyword-only arguments of the two decompositions.
static bool
decomposition_args(PyObject* py_lowrank, PyObject* py_max_iters, PyObject* py_threshold,
                   int64_t& lowrank, int64_t& max_iters, double& threshold)
{
    return m00nny::pyarg::to_int(py_lowrank, "lowrank", -1, lowrank)
        && m00nny::pyarg::to_int(py_max_iters, "max_iters", m00nny::kEigenMaxIters, max_iters)
        && m00nny::pyarg::to_double(py_threshold, "threshold", m00nny::kEigenThreshold, threshold);
}

PyObject*
linalg_eigen_decomposition
(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", "lowrank", "max_iters", "threshold", NULL};
    PyObject *pyX = nullptr, *py_lowrank = nullptr, *py_max_iters = nullptr, *py_threshold = nullptr;
    torch::Tensor X;
    int64_t lowrank, max_iters;
    double threshold;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOO", (char**)keywords, &pyX, &py_lowrank, &py_max_iters, &py_threshold)) return NULL;
    if (!m00nny::pyarg::to_tensor(pyX, "X", X)
        || !decomposition_args(py_lowrank, py_max_iters, py_threshold, lowrank, max_iters, threshold)) return NULL;
    LIBM00NNY_PY_TRY
    {
        auto [L, V] = m00nny::eigen_decomposition(X, lowrank, max_iters, threshold);
        return Py_BuildValue("(NN)", THPVariable_Wrap(L), THPVariable_Wrap(V));
    }
    LIBM00NNY_PY_CATCH(NULL)
}

PyObject*
linalg_singular_value_decomposition
(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", "lowrank", "max_iters", "threshold", NULL};
    PyObject *pyX = nullptr, *py_lowrank = nullptr, *py_max_iters = nullptr, *py_threshold = nullptr;
    torch::Tensor X;
    int64_t lowrank, max_iters;
    double threshold;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOO", (char**)keywords, &pyX, &py_lowrank, &py_max_iters, &py_threshold)) return NULL;
    if (!m00nny::pyarg::to_tensor(pyX, "X", X)
        || !decomposition_args(py_lowrank, py_max_iters, py_threshold, lowrank, max_iters, threshold)) return NULL;
    LIBM00NNY_PY_TRY
    {
        auto [U, S, V] = m00nny::singular_value_decomposition(X, lowrank, max_iters, threshold);
        return Py_BuildValue("(NNN)", THPVariable_Wrap(U), THPVariable_Wrap(S), THPVariable_Wrap(V));
    }
    LIBM00NNY_PY_CATCH(NULL)
}

PyMODINIT_FUNC
PyInit_linalg
(void)
{
    PyObject *m;
    m = LibM00nnyCreateModule(&Linalg_Module);
    if (m == NULL)
        return NULL;
    return m;
}
