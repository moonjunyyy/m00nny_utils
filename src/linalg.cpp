#include "linalg.h"

torch::Tensor
m00nny::kronecker_product(torch::Tensor& X, torch::Tensor& Y)
{
    /*
    Kronecker Product of two tensors.
    This assumes the output tensor is a 4D tensor, not actually a Kronecker product, Tensor product.
    The output tensor is the element-wise multiplication of the two tensors.

    Args:
        X (torch::Tensor): The first input tensor.  (*, n, m)
        Y (torch::Tensor): The second input tensor. (*, p, q)

    Returns:
        torch::Tensor: The output tensor. (*, n, p, m, q)
    */
    int64_t dim_X = X.dim();
    int64_t dim_Y = Y.dim();

    if (dim_X != dim_Y || dim_X < 2 || dim_Y < 2)
    {
        LIBM00NNY_ERROR_MESSAGE("The input tensors must have the same number of dimensions greater than 1.\n")
        return torch::Tensor();
    }

    c10::IntArrayRef size_X = X.sizes();
    c10::IntArrayRef size_Y = Y.sizes();

    int64_t n = size_X[dim_X - 2], m = size_X[dim_X - 1];
    int64_t p = size_Y[dim_Y - 2], q = size_Y[dim_Y - 1];

    int64_t num_matrix = 1;
    for (int64_t i = 0; i < dim_X; i++)
    {
        if (i < dim_X - 2)
        {
            num_matrix *= size_X[i];
            if (size_X[i] != size_Y[i])
            {
                LIBM00NNY_ERROR_MESSAGE("The input tensors must have the same shape except for the last two dimensions.\n")
                return torch::Tensor();
            }
        }
    }

    torch::Tensor X_reshape = X.view({num_matrix, n, 1, m, 1});
    torch::Tensor Y_reshape = Y.view({num_matrix, 1, p, 1, q});

    auto target_size = X_reshape.sizes().vec();
    target_size[dim_X - 4] = n;
    target_size[dim_X - 3] = p;
    target_size[dim_X - 2] = m;
    target_size[dim_X - 1] = q;

    torch::Tensor Out = torch::empty(target_size, X.options());
    torch::Tensor Out_reshape = Out.view({num_matrix, n, p, m, q});
    torch::mul_out(Out_reshape, X_reshape, Y_reshape);
    return Out;
}

torch::Tensor
m00nny::inverse(torch::Tensor& X)
{
    /*
    Get the inverse matrix of given tensor X.
    This function assumes that the input tensor is a square matrix.
    This function uses the Gauss-Jordan Elimination Method, takes little more time, getting little more accurate.

    Args:
        X (torch::Tensor): The input tensor. (*, n, n)

    Returns:
        torch::Tensor: The output tensor. (*, n, n)
    */
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::inverse\n")
    c10::IntArrayRef size = X.sizes();
    int64_t dim = X.dim();
    int64_t n   = size[dim - 2];
    int64_t m   = size[dim - 1];
    // If the input tensor is not a square matrix, return the input tensor.
    int64_t num_matrix = 1;
    for (int64_t i = 0; i < dim - 2; i++) num_matrix *= size[i];
    
    // Make Identity Matrix with the same shape as the input tensor.
    // Use the Gauss-Jordan Elimination Method to calculate the inverse matrix.
    torch::Tensor I = torch::eye(n, m, X.options()).unsqueeze(0).repeat({num_matrix, 1, 1});
    torch::Tensor A = torch::cat({X.view({num_matrix, n, m}), I.view({num_matrix, n, m})}, 2); // A : (*, n, 2m) where n = m. A is the augmented matrix.
    torch::Tensor num_arange = torch::arange(0, num_matrix, 1, X.options().dtype(torch::kLong)); // indexing tensor.
    torch::Tensor arange     = torch::arange(0, n, 1, X.options().dtype(torch::kLong)).unsqueeze(0).repeat({num_matrix, 1}); // rearrange buffer for row swapping.
    torch::Tensor _buffer    = torch::empty_like(A); // buffer for row elimination.
    torch::Tensor _buffer_flatten = _buffer.view({num_matrix * n * 2 * m}); // buffer for row elimination.
    torch::Tensor _temp     = torch::empty({num_matrix}, X.options().dtype(torch::kLong)); // Temporary tensor for row swapping.
    torch::Tensor max_idx   = torch::empty({num_matrix}, X.options().dtype(torch::kLong)); // buffer for getting the index of the maximum value.
    torch::Tensor max_value = torch::empty({num_matrix}, X.options()); // buffer for getting the maximum value.
    torch::Tensor Aslice;
    torch::Tensor _temp_index;
    torch::Tensor _buffer_vector;
    for (int64_t i = 0; i < n; i++)
    {
        // Select the row with maximum value in the i-th column.
        // max calculation is O(n).
        _buffer_vector = _buffer_flatten.slice(0, 0, num_matrix * (n-i));
        _buffer_vector = _buffer_vector.view({num_matrix, n-i});
        _temp_index = arange.slice(1, i, n, 1);
        _temp_index = _temp_index.view({num_matrix, n-i});
        Aslice = A.index({num_arange.unsqueeze(1), _temp_index, i}).view({num_matrix, n-i});
        torch::abs_out(_buffer_vector, Aslice);
        torch::argmax_out(_temp, _buffer_vector, 1); // max_idx : (*)
        torch::add_out(max_idx, _temp, i); // max_idx : (*)

        // Swap the rows. swap is O(1).
        _temp_index = arange.index({num_arange, max_idx}); // _temp_index : (*)
        _temp.copy_(arange.index({num_arange, i}));        // _temp : (*)
        arange.index_put_({num_arange, i}, _temp_index);   // arange : (*, n)
        arange.index_put_({num_arange, max_idx}, _temp);   // arange : (*, n)

        max_value.copy_(A.index({num_arange, _temp_index, i})); // max_value : (*)
        _buffer.fill_(1); // _buffer : (*, n, 2m)
        _buffer.index_put_({num_arange, _temp_index}, max_value); // _buffer : (*, n, 2m)
        _buffer.reciprocal_();
        A.mul_(_buffer); // A : (*, n, 2m)
        
        // Eliminate the i-th column.
        // eliminate is O(n^2).
        torch::mul_out(_buffer, A.index({num_arange, _temp_index}).unsqueeze(1),
                                A.slice(2, i, i+1, 1)); // (*, 1, 2m) * (*, n, 1) = (n, 2m)
        _buffer.index_put_({num_arange, _temp_index}, 0); // (*, n, 2m)
        A.sub_(_buffer); // (n, 2m) - (n, 2m) = (n, 2m)
    }
    // Total complexity is O(n^3).
    // Now, the left side of the matrix A is the identity matrix, and the right side is the inverse matrix.
    A = A.slice(2, n, 2*n, 1); // A : (*, n, m) where n = m. A is the inverse matrix.
    A = A.index({num_arange.repeat_interleave(n), arange.view({-1})}).reshape(size); // A : (*, n, m) where n = m. A is the inverse matrix.
    return A;
}

torch::Tensor
m00nny::orthogonalize(torch::Tensor& X)
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
    c10::IntArrayRef size = X.sizes();
    int64_t dim = X.dim();
    int64_t n   = size[dim - 2];
    int64_t m   = size[dim - 1];
    // If the input tensor is not a square matrix, return the input tensor.
    int64_t num_matrix = 1;
    for (int64_t i = 0; i < dim - 2; i++) num_matrix *= size[i];
    
    torch::Tensor A = X.view({num_matrix, n, m}).clone(); // A : (*, n, m) A is the matrix to otthogonalize.
    A.div_(A.norm(2, 2, true)); // A : (*, n, m) A is the normalized matrix.
    torch::Tensor cumsum_A = A.cumsum(1) - A; // cumsum_A : (*, n, m) cumsum_A is the cumulative sum of the matrix.
    cumsum_A.slice(1, 0, 1, 1) = 1;
    cumsum_A.div_(cumsum_A.norm(2, 2, true)); // normalized_cumsum_A : (*, n, m) normalized_cumsum_A is the normalized cumulative sum of the matrix.
    cumsum_A.slice(1, 0, 1, 1) = 0;
    A.sub_(A.mul(cumsum_A).sum(2, true).mul(cumsum_A)); // A : (*, n, m) A is the orthogonalized matrix.
    A.div_(A.norm(2, 2, true)); // A : (*, n, m) A is the orthogonalized matrix.
    return A.reshape(size);
}

std::tuple<torch::Tensor, torch::Tensor>
m00nny::eigen_decomposition(torch::Tensor& X, int64_t lowrank, int64_t max_iters, double threshold)
{
    /*
    Egien Decomposition of the input tensor.
    This function uses the Newton's Method to calculate the eigenvectors.
    This takes more time than the eigendecomposition of the PyTorch library, but it is considerably more accurate. (Especially for the eigenvectors with small eigenvalues.)

    Args:
        X (torch::Tensor): The input tensor. (*, n, m) where n = m (Square Matrix)
        lowrank (int64_t): The number of eigenvectors to calculate. If lowrank is less than 0 or greater than n, it is set to n.
        max_iters (int64_t): The maximum number of iterations. If the error is less than the threshold, the iteration stops.
        threshold (double): The threshold to stop the iteration.

    Returns:
        std::tuple<torch::Tensor, torch::Tensor>: The eigenvalue tensor and the eigenvector tensor. (*, n, n) and (*, n, n)
    */
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::eigen_decomposition\n")
    torch::NoGradGuard no_grad;
    // X: (*, n, m) where n = m (Square Matrix)
    c10::IntArrayRef size = X.sizes();
    int64_t dim = X.dim();
    int64_t n   = size[dim - 2];
    int64_t m   = size[dim - 1];
    int64_t num_matrix = 1;
    for (int64_t i = 0; i < dim - 2; i++) num_matrix *= size[i];

    // If the input tensor is not a square matrix, return the input tensor.
    lowrank = (lowrank < 0) || (lowrank > n) ? n : lowrank;
            
    torch::Tensor I_d = torch::eye(n, m, X.options());             // I : (n, n). I is the identity matrix.
    I_d.unsqueeze_(0); // I : (*, n, n). I is the identity matrix.
    torch::Tensor I_r = I_d.slice(1, 0, lowrank, 1).slice(2, 0, lowrank, 1); // Ir : (*, r, r). Ir is the identity matrix, sharing the same memory with I.

    // Initialize the eigenvector matrix.
    // torch::Tensor A  = X.clone(); // A : (*, n, n) where n = m. A is the matrix to calculate the eigenvectors.
    torch::Tensor A   = X.view({num_matrix, n, m}).clone();    // A         : (*, n, m) where n = m. A is the matrix to calculate the eigenvectors.
    torch::Tensor At  = A.transpose(1, 2).clone();             // At        : (*, m, n) where n = m. At is the transposed matrix to calculate the eigenvectors.
    torch::Tensor A_lowrank = A.slice(1, 0, lowrank, 1);       // A_lowrank : (*, r, m) where n = m. A_lowrank is the lowrank matrix to calculate the eigenvectors.
    torch::Tensor AtA = torch::matmul(At, At.transpose(1, 2)).flatten().view({num_matrix, n, m}); // AtA       : (*, m, m) where n = m. AtA is the matrix to calculate the eigenvectors.

    torch::Tensor Vt = torch::randn_like(A_lowrank);  // Vt : (*, r, n) V is the eigenvector matrix.
    Vt = orthogonalize(Vt); // V : (*, n, r). V is the orthogonalized eigenvector matrix.
    torch::Tensor V  = Vt.transpose(1, 2).clone();    // V  : (*, n, r) V is the eigenvector matrix.

    torch::Tensor Λ  = torch::zeros({num_matrix, lowrank * lowrank}, X.options()).view({num_matrix, lowrank, lowrank}); // Λ  : (*, r, r). Λ is the eigenvalue matrix.
    torch::Tensor vec_Λ  = Λ.view({num_matrix, lowrank * lowrank}).slice(1, 0, lowrank * lowrank, lowrank + 1); // vec_Λ  : (*, r). vec_Λ is the eigenvalue vector.

    torch::Tensor AV  = torch::empty_like(V);  // AV  : (*, n, r). AV is the matrix-vector multiplication.
    torch::Tensor ΛV  = torch::empty_like(V);  // ΛV  : (*, n, r). ΛV is the matrix-vector multiplication.
    torch::Tensor E   = torch::empty({num_matrix * n * lowrank}, X.options()).view_as(V);  // E   : (*, n, r). E is the error matrix.
    torch::Tensor Et  = torch::empty({num_matrix * n * lowrank}, X.options()).view_as(Vt); // Et  : (*, r, n). Et is the transposed error matrix.

    torch::Tensor U  = torch::empty_like(V);  // U  : (*, n, r). U is the matrix to calculate the eigenvectors.
    torch::Tensor Ut = torch::empty_like(Vt); // Ut : (*, r, n). Ut is the transposed matrix to calculate the eigenvectors.

    torch::Tensor AtU = torch::empty_like(V);  // At U : (*, n, r).
    torch::Tensor VVt = torch::empty({num_matrix * n * n}, X.options()).view({num_matrix, n, n});                         // VVt : (*, n, n).
    torch::Tensor VtV = torch::empty({num_matrix * lowrank * lowrank}, X.options()).view({num_matrix, lowrank, lowrank}); // VtV : (*, r, r).
    torch::Tensor VtE = torch::empty({num_matrix * lowrank * lowrank}, X.options()).view({num_matrix, lowrank, lowrank}); // Vt E : (*, r, r).

    torch::Tensor dE_dV = torch::empty({num_matrix * n * lowrank * n * lowrank}, X.options()).view({num_matrix, n, lowrank, n, lowrank}); // dE_dV : (*, n*r, n*r). dE_dV is the gradient of the eigenvalue vector and the eigenvector matrix.
    torch::Tensor vec_dE_dV = dE_dV.view({num_matrix, n * lowrank, n * lowrank}); // dE_dV : (*, n*r, n*r). dE_dV is the gradient of the eigenvalue vector and the eigenvector matrix.
    
    torch::Tensor dL1_dV = torch::empty({num_matrix, n * lowrank}, X.options()).view({num_matrix, n, lowrank}); // dL1_dV : (*, n, r). gL1_gV is the gradient of the eigenvalue vector and the eigenvector matrix.
    torch::Tensor dL2_dV = torch::empty({num_matrix, n * lowrank}, X.options()).view({num_matrix, n, lowrank}); // dL2_dV : (*, n, r). gL2_gV is the gradient of the eigenvalue vector and the eigenvector matrix.
    torch::Tensor vec_dL1_dV = dL1_dV.view({num_matrix, n * lowrank, 1}); // vec_dL1_dV : (*, n*r, 1).
    torch::Tensor vec_dL2_dV = dL2_dV.view({num_matrix, n * lowrank, 1}); // vec_dL2_dV : (*, n*r, 1).
    
    torch::Tensor ddL1_dVdVt = torch::empty({num_matrix, n * lowrank * lowrank * n}, X.options()).view({num_matrix, n * lowrank, lowrank * n}); // dL1_dVdVt : (*, n, n). ggL1_gVgVt is the second derivative.
    torch::Tensor ddL2_dVdVt = torch::empty({num_matrix, n * lowrank * lowrank * n}, X.options()).view({num_matrix, n * lowrank, lowrank * n}); // dL2_dVdVt : (*, n, n). ggL2_gVgVt is the second derivative.

    torch::Tensor _buffer_0 = torch::empty({num_matrix * n * lowrank * n * lowrank}, X.options()); // temporary buffer for calculation.
    torch::Tensor _buffer_0_nrnr = _buffer_0.view({num_matrix, n, lowrank, n, lowrank});
    torch::Tensor _buffer_0_rnnr = _buffer_0.view({num_matrix, lowrank, n, n, lowrank});
    torch::Tensor _buffer_0_nrrn = _buffer_0.view({num_matrix, n, lowrank, lowrank, n});
    torch::Tensor _buffer_0_rnrn = _buffer_0.view({num_matrix, lowrank, n, lowrank, n});
    torch::Tensor _buffer_0_like_A = _buffer_0.slice(0, 0, num_matrix * n * n).view({num_matrix, n, n});
    torch::Tensor _buffer_0_like_vec_V = _buffer_0.slice(0, 0, num_matrix * n * lowrank).view({num_matrix, n * lowrank, 1});
    torch::Tensor _buffer_0_like_Vt = _buffer_0_like_vec_V.view({num_matrix, lowrank, n});
    torch::Tensor _buffer_0_like_V = _buffer_0_like_vec_V.view({num_matrix, n, lowrank});
    
    torch::Tensor _buffer_1 = torch::empty({num_matrix * n * lowrank * n * lowrank}, X.options()); // temporary buffer for calculation.
    torch::Tensor _buffer_1_nrnr = _buffer_1.view({num_matrix, n, lowrank, n, lowrank});
    torch::Tensor _buffer_1_rnnr = _buffer_1.view({num_matrix, lowrank, n, n, lowrank});
    torch::Tensor _buffer_1_nrrn = _buffer_1.view({num_matrix, n, lowrank, lowrank, n});
    torch::Tensor _buffer_1_rnrn = _buffer_1.view({num_matrix, lowrank, n, lowrank, n});
    torch::Tensor _buffer_1_like_A = _buffer_1.slice(0, 0, num_matrix * n * n).view({num_matrix, n, n});
    torch::Tensor _buffer_1_like_vec_V = _buffer_1.slice(0, 0, num_matrix * n * lowrank).view({num_matrix, n * lowrank, 1});
    torch::Tensor _buffer_1_like_Vt = _buffer_1_like_vec_V.view({num_matrix, lowrank, n});
    torch::Tensor _buffer_1_like_V = _buffer_1_like_vec_V.view({num_matrix, n, lowrank});

    for (int64_t j = 0; j < max_iters; j++)
    {
        torch::norm_out(vec_Λ, V, 2, 1);                // Λ  : (*, r). Λ is the Eigenvalue vector.
        V.div_(vec_Λ.unsqueeze(1));                     // V  : (*, n, r). V is the normalized eigenvector matrix.
        torch::transpose_copy_out(Vt, V, 1, 2);         // Vt : (*, r, n). Vt is the transposed eigenvector matrix.

        torch::matmul_out(AV, A, Vt.transpose(1, 2));   // AV : (*, n, r). AV is the matrix-vector multiplication.
        torch::norm_out(vec_Λ, AV, 2, 1);               // Λ  : (*, r). Λ is the Eigenvalue vector.
        torch::mul_out(ΛV, V, vec_Λ.unsqueeze(1));      // ΛV : (*, n, r). ΛV is the Eigenvalue Vector multiplication.
        torch::sub_out(E, AV, ΛV);                      // E  : (*, n, m). E is the error matrix.
        torch::transpose_copy_out(Et, E, 1, 2);         // Et : (*, m, n). Et is the transposed error matrix.
        if (E.abs().max().item<float>() < threshold) break; // If the error is less than the threshold, break the loop.

        torch::div_out(U, AV, vec_Λ.unsqueeze(1));      // U  : (*, n, m). U is the matrix to calculate the eigenvectors.
        torch::transpose_copy_out(Ut, U, 1, 2);         // Ut : (*, m, n). Ut is the transposed matrix to calculate the eigenvectors.

        torch::matmul_out(AtU, At, Ut.transpose(1, 2)); // At U
        torch::matmul_out(VtE, Vt, Et.transpose(1, 2)); // Vt E

        torch::matmul_out(_buffer_1_like_A, AtU, V.transpose(1, 2));  // At U Vt
        torch::sub_out(_buffer_0_like_A, At, _buffer_1_like_A);       // At - At U Vt
        torch::sub_out(dE_dV,
            _buffer_0_like_A.view({num_matrix, n, 1, n, 1}),
            Λ.view({num_matrix, 1, lowrank, 1, lowrank}));            // dE_dV : (*, n, r, n, r).

        torch::matmul_out(vec_dL1_dV,
            dE_dV.view({num_matrix, n * lowrank, n * lowrank}),
            E.view({num_matrix, n * lowrank, 1}));                    // vec_dL1_dV : (*, n*r, 1).

        _buffer_0_nrrn.copy_(dE_dV.permute({0, 1, 2, 4, 3}));         // (*, n, r, r, n)
        _buffer_1_rnrn.copy_(dE_dV.permute({0, 2, 1, 3, 4}));         // (*, r, n, r, n)
        torch::matmul_out(ddL1_dVdVt,
            _buffer_0.view({num_matrix, n * lowrank, n * lowrank}),
            _buffer_1.view({num_matrix, n * lowrank, n * lowrank}));   // ddL1_dVdVt : (*, n*r, r*n).
        torch::mul_out(_buffer_0_nrrn,
            E.view({num_matrix, n, 1, lowrank, 1}),
            AtU.transpose(1, 2).view({num_matrix, 1, lowrank, 1, n}));                  // E x Ut A (*, n, r, r, n)
        _buffer_1_nrrn.copy_(_buffer_0_nrrn.permute({0, 4, 3, 2, 1}));
        ddL1_dVdVt.add_(_buffer_0_nrrn.view({num_matrix, n * lowrank, lowrank * n}));   // ddL1_dVdVt : (*, n*r, r*n).
        ddL1_dVdVt.add_(_buffer_1_nrrn.view({num_matrix, n * lowrank, lowrank * n}));   // ddL1_dVdVt : (*, n*r, r*n).
        
        torch::div_out(_buffer_1_like_V, AtU, vec_Λ.unsqueeze(1));                      // AtU Λ^-1
        torch::matmul_out(_buffer_0_like_A, _buffer_1_like_V, AtU.transpose(1, 2));     // AtU Λ^-1 Ut A
        torch::mul_out(_buffer_1_rnrn,
            VtE.view({num_matrix, lowrank, 1, lowrank, 1}),
            _buffer_0_like_A.view({num_matrix, 1, n, 1, n}));                           // Vt E x AtU Λ^-1 Ut A : (*, r, n, r, n)
        torch::permute_copy_out(_buffer_0_nrrn, _buffer_1_rnrn, {0, 2, 1, 3, 4});       // (*, n, r, r, n)
        ddL1_dVdVt.add_(_buffer_0_nrrn.view({num_matrix, n * lowrank, lowrank * n}));   // ggL1_gVgVt : (*, n*r, r*n).

        VtE.div_(vec_Λ.unsqueeze(2));
        torch::mul_out(_buffer_0_rnrn,
            VtE.view({num_matrix, lowrank, 1, lowrank, 1}),
            AtA.view({num_matrix, 1, n, 1, n}));                                        // Λ^-1 Vt E x AtA : (*, r, n, r, n)
        torch::permute_copy_out(_buffer_0_nrrn, _buffer_1_rnrn, {0, 2, 1, 3, 4});       // (*, n, r, r, n)
        ddL1_dVdVt.add_(_buffer_0_nrrn.view({num_matrix, n * lowrank, lowrank * n}));   // ggL1_gVgVt : (*, n*r, r*n).

        torch::matmul_out(VtV, Vt, Vt.transpose(1, 2)); // VtV : (*, r, r). VtV is the inner product of the eigenvector matrix.
        torch::matmul_out(VVt, V, V.transpose(1, 2));   // VVt : (*, n, n). VVt is the inner product of the eigenvector matrix.

        // Get the gradient of the inner product of the eigenvector matrix. (Cosine Similarity)
        torch::matmul_out(dL2_dV, V, VtV.transpose(1,2)); // gL2_gV : (*, n, r). gL2_gV is the gradient of the inner product of the eigenvector matrix.
        dL2_dV.sub_(V);
        
        torch::mul_out(_buffer_0_nrrn,
            V.view({num_matrix, n, 1, lowrank, 1}),
            Vt.view({num_matrix, 1, lowrank, 1, n}));                                  // ggL2_gVgVt : (*, n, r, r, n).
        ddL2_dVdVt.copy_(_buffer_0_nrrn.view({num_matrix, n * lowrank, lowrank * n})); // ggL2_gVgVt : (*, n*r, r*n).
        
        torch::mul_out(_buffer_0_rnrn,
            VtV.view({num_matrix, lowrank, 1, lowrank, 1}),
            I_d.view({1, 1, n, 1, n}));                                                // (*, r, n, r, n)
        torch::permute_copy_out(_buffer_1_nrrn, _buffer_0_rnrn, {0, 2, 1, 3, 4});      // (*, n, r, r, n)
        ddL2_dVdVt.add_(_buffer_1_nrrn.view({num_matrix, n * lowrank, lowrank * n}));  // ggL2_gVgVt : (*, n*r, r*n).

        torch::mul_out(_buffer_0_rnrn,
            I_r.view({num_matrix, lowrank, 1, lowrank, 1}),
            VVt.view({1, 1, n, 1, n}));                                                // (*, r, n, r, n)
        torch::permute_copy_out(_buffer_1_nrrn, _buffer_0_rnrn, {0, 2, 1, 3, 4});      // (*, n, r, r, n)
        ddL2_dVdVt.add_(_buffer_1_nrrn.view({num_matrix, n * lowrank, lowrank * n}));  // ggL2_gVgVt : (*, n*r, r*n).

        torch::mul_out(_buffer_0_rnrn,
            I_r.view({num_matrix, lowrank, 1, lowrank, 1}),
            I_d.view({1, 1, n, 1, n}));                                                // (*, r, n, r, n)
        torch::permute_copy_out(_buffer_1_nrrn, _buffer_0_rnrn, {0, 2, 1, 3, 4});      // (*, n, r, r, n)
        ddL2_dVdVt.sub_(_buffer_1_nrrn.view({num_matrix, n * lowrank, lowrank * n}));  // ggL2_gVgVt : (*, n*r, r*n).

        // Update the V based on the gradient.
        _buffer_0_like_Vt.copy_(dL1_dV.transpose(1, 2)); // _buffer_0_like_Vt : (*, r, n).
        torch::matmul_out(_buffer_1_like_vec_V, inverse(ddL1_dVdVt), _buffer_0_like_vec_V); // _buffer_1 = ggL1_gVgVt^-1 gL1_gV
        V.sub_(_buffer_1_like_V); // V = V - _buffer_1

        _buffer_0_like_Vt.copy_(dL2_dV.transpose(1, 2)); // _buffer_0_like_Vt : (*, r, n).
        torch::matmul_out(_buffer_1_like_vec_V, inverse(ddL2_dVdVt), _buffer_0_like_vec_V); // _buffer_2 = ggL2_gVgVt^-1 gL2_gV
        V.sub_(_buffer_1_like_V); // V = V - _buffer_1
    }
    V = V.reshape(size);
    return std::make_tuple(Λ, V);
}

std::tuple<torch::Tensor, torch::Tensor>
m00nny::eigen_decomposition_new(torch::Tensor& X, int64_t lowrank, int64_t max_iters, double threshold)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::eigen_decomposition\n")
    torch::NoGradGuard no_grad;
    // X: (*, n, m) where n = m (Square Matrix)
    c10::IntArrayRef size = X.sizes();
    int64_t dim = X.dim();
    int64_t n   = size[dim - 2];
    int64_t m   = size[dim - 1];
    int64_t num_matrix = 1;
    for (int64_t i = 0; i < dim - 2; i++) num_matrix *= size[i];

    // If the input tensor is not a square matrix, return the input tensor.
    lowrank = (lowrank < 0) || (lowrank > n) ? n : lowrank;
    
    // Initialize the eigenvector matrix.
    // torch::Tensor A  = X.clone(); // A : (*, n, n) where n = m. A is the matrix to calculate the eigenvectors.
    torch::Tensor A  = X.view({num_matrix, n, m}).clone(); // A         : (*, n, m) where n = m. A is the matrix to calculate the eigenvectors.
    torch::Tensor At = A.transpose(1, 2).clone();          // At        : (*, m, n) where n = m. At is the transposed matrix to calculate the eigenvectors.
    torch::Tensor A_lowrank = A.slice(1, 0, lowrank, 1);   // A_lowrank : (*, r, m) where n = m. A_lowrank is the lowrank matrix to calculate the eigenvectors.

    torch::Tensor Vt = torch::randn_like(A_lowrank);  // Vt : (*, r, n) V is the eigenvector matrix.
    torch::Tensor V  = Vt.transpose(1, 2).clone();    // V  : (*, n, r) V is the eigenvector matrix.
    torch::Tensor Λ  = torch::zeros({num_matrix, n, n},
                                        X.options()); // Λ  : (*, n, n). Λ is the eigenvalue vector.
    torch::Tensor Λr = Λ
        .slice(1, 0, lowrank, 1)
        .slice(2, 0, lowrank, 1);                     // Λr : (*, r). Λr is the eigenvalue vector, sharing the same memory with Λ.

    torch::Tensor vec_Λ  = torch::diagonal(Λ,  0, 1, 2); // vec_Λ  : (*, n). vec_Λ is the eigenvalue vector.
    torch::Tensor vec_Λr = torch::diagonal(Λr, 0, 1, 2); // vec_Λr : (*, r). vec_Λr is the eigenvalue vector, sharing the same memory with vec_Λ.

    torch::Tensor AV  = torch::empty_like(V);  // AV  : (*, n, r). AV is the matrix-vector multiplication.
    torch::Tensor VΛ  = torch::empty_like(V);  // VΛ  : (*, n, r). ΛV is the matrix-vector multiplication.
    torch::Tensor E   = torch::empty_like(V);  // E   : (*, n, r). E is the error matrix.
    torch::Tensor Et  = torch::empty_like(Vt); // Et  : (*, r, n). Et is the transposed error matrix.
    torch::Tensor VVt = torch::empty_like(A);  // VVt : (*, n, n). VVt is the inner product of the eigenvector matrix.
    torch::Tensor VtV = torch::empty({num_matrix, lowrank, lowrank}, X.options()); // VtV : (*, r, r). VtV is the inner product of the eigenvector matrix.

    torch::Tensor I = torch::eye(n, m, X.options());             // I : (n, n). I is the identity matrix.
    for (int64_t d = 0; d < (dim == 2 ? 1 : dim - 2); d++) I.unsqueeze_(0); // I : (*, n, n). I is the identity matrix.
    torch::Tensor Ir = I.slice(1, 0, lowrank, 1).slice(2, 0, lowrank, 1); // Ir : (*, r, r). Ir is the identity matrix, sharing the same memory with I.

    torch::Tensor gL1_gV = torch::empty_like(V);     // gL1_gV : (*, n, r). gL1_gV is the gradient of the eigenvalue vector and the eigenvector matrix.
    torch::Tensor gL2_gV = torch::empty_like(V);     // gL2_gV : (*, n, r). gL2_gV is the gradient of the eigenvalue vector and the eigenvector matrix.
    torch::Tensor ggL1_gVgVt = torch::empty_like(A); // ggL1_gVgVt : (*, n, n). ggL1_gVgVt is the second derivative of the eigenvalue vector and the eigenvector matrix and the eigenvector matrix transpose.
    torch::Tensor ggL2_gVgVt = torch::empty_like(A); // ggL2_gVgVt : (*, n, n). ggL2_gVgVt is the second derivative of the eigenvalue vector and the eigenvector matrix and the eigenvector matrix transpose.

    torch::Tensor _buffer_1 = torch::empty_like(A); // temporary buffer for calculation.
    torch::Tensor _buffer_2 = torch::empty_like(A); // temporary buffer for calculation.
    torch::Tensor _buffer_3 = torch::empty_like(A); // temporary buffer for calculation.
    
    torch::Tensor _buffer_1_like_V = _buffer_1.view({-1}).slice(0, 0, num_matrix * n * lowrank).view({num_matrix, n, lowrank}); // temporary buffer for calculation.
    torch::Tensor _buffer_2_like_V = _buffer_2.view({-1}).slice(0, 0, num_matrix * n * lowrank).view({num_matrix, n, lowrank}); // temporary buffer for calculation.
    torch::Tensor _buffer_3_like_V = _buffer_3.view({-1}).slice(0, 0, num_matrix * n * lowrank).view({num_matrix, n, lowrank}); // temporary buffer for calculation.

    torch::Tensor _buffer_1_like_R = _buffer_1.view({-1}).slice(0, 0, num_matrix * lowrank * lowrank).view({num_matrix, lowrank, lowrank}); // temporary buffer for calculation.
    torch::Tensor _buffer_2_like_R = _buffer_2.view({-1}).slice(0, 0, num_matrix * lowrank * lowrank).view({num_matrix, lowrank, lowrank}); // temporary buffer for calculation.
    torch::Tensor _buffer_3_like_R = _buffer_3.view({-1}).slice(0, 0, num_matrix * lowrank * lowrank).view({num_matrix, lowrank, lowrank}); // temporary buffer for calculation.

    for (int64_t j = 0; j < max_iters; j++)
    {
        V.div_(V.norm(2, 1, true)); // Normalize the eigenvector matrix.
        torch::transpose_copy_out(Vt, V, 1, 2); // Vt : (*, r, n). Vt is the transposed eigenvector matrix.

        torch::bmm_out(AV, A, Vt.transpose(1, 2)); // AV : (*, n, r). AV is the matrix-vector multiplication.
        torch::norm_out(vec_Λr, AV, 2, 1);            // Λr : (*, r). Λ is the Eigenvalue vector.
        // torch::mul_out(ΛV, V, vec_Λr.unsqueeze(1));   // ΛV : (*, n, r). ΛV is the Eigenvalue Vector multiplication.
        torch::bmm_out(VΛ, V, Λr);   // ΛV : (*, n, r). ΛV is the Eigenvalue Vector multiplication.
        torch::sub_out(E, AV, VΛ);                    // E  : (*, n, m). E is the error matrix.
        torch::transpose_copy_out(Et, E, 1, 2);       // Et : (*, m, n). Et is the transposed error matrix.
        if (E.abs().max().item<float>() < threshold) break; // If the error is less than the threshold, break the loop.

        // Get the gradient of Loss_1, Squared Sum Error.
        torch::sub_out(_buffer_1, At, Λ); // (A - Λ)^T = (A^T - Λ)
        torch::bmm_out(_buffer_3_like_V, _buffer_1, Et.transpose(1, 2)); // (A - Λ)^T E
        torch::mul_out(_buffer_2_like_V, E, vec_Λr.unsqueeze(1)); //  E Λ
        torch::sub_out(gL1_gV, _buffer_3_like_V, _buffer_2_like_V).mul_(2); // gL1_gV = 2 [(A - Λ)^T E - E Λ]

        // torch::mul_out(_buffer_2, _buffer_1, vec_Λ.unsqueeze(2)).mul_(-2); // -2 Λ (A - Λ)^T
        // torch::bmm_out(_buffer_2, Λ, _buffer_1).mul_(-2); // -2 Λ (A - Λ)^T
        torch::bmm_out(_buffer_2, _buffer_1, Λ.transpose(1, 2)).mul_(-2); // -2 (A - Λ) Λ
        // torch::sub_out(_buffer_3, _buffer_1, Λ).mul_(-1); // (A^T - 2Λ) = (A - 2Λ)^T
        torch::sub_out(_buffer_3, _buffer_1, Λ).mul_(-1); // (2Λ - A^T) = (2Λ - A)^T
        torch::bmm_out(_buffer_1, _buffer_3, _buffer_3.transpose(1, 2)); // (A - 2Λ)^T (A - 2Λ)
        torch::add_out(ggL1_gVgVt, _buffer_1, _buffer_2).mul_(2); // ggL1_gVgVt 2 [(A - 2Λ)^T (A - 2Λ) - 2 (A - Λ) Λ]

        // Get the gradient of the inner product of the eigenvector matrix. (Cosine Similarity)
        torch::bmm_out(_buffer_1_like_R, Vt, Vt.transpose(1, 2)); // VtV : (*, r, r) where n = m. VtV is the inner product of the eigenvector matrix.
        _buffer_1_like_R.sub_(Ir); // VtV - Ir
        torch::bmm_out(gL2_gV, V, _buffer_1_like_R.transpose(1, 2)); // gL2_gV = V (VtV - Ir) : (*, n, r)
        gL2_gV.mul_(4);

        torch::bmm_out(ggL2_gVgVt, V, Vt);
        ggL2_gVgVt.mul_(3);
        ggL2_gVgVt.sub_(I);
        ggL2_gVgVt.mul_(4); // ggL2_gVgVt = 12 V Vt - 4 I : (*, n, n)

        // Update the V based on the gradient.
        torch::matmul_out(_buffer_1_like_V, inverse(ggL1_gVgVt), gL1_gV); // _buffer_1 = ggL1_gVgVt^-1 gL1_gV
        torch::matmul_out(_buffer_2_like_V, inverse(ggL2_gVgVt), gL2_gV); // _buffer_2 = ggL2_gVgVt^-1 gL2_gV

        // torch::matmul_out(_buffer_1_like_V, torch::inverse(ggL1_gVgVt), gL1_gV); // _buffer_1 = ggL1_gVgVt^-1 gL1_gV
        // torch::matmul_out(_buffer_2_like_V, torch::inverse(ggL2_gVgVt), gL2_gV); // _buffer_2 = ggL2_gVgVt^-1 gL2_gV

        V.sub_(_buffer_1_like_V).sub_(_buffer_2_like_V); // V = V - 0.5 (_buffer_1 + _buffer_2)
    }
    Λ = Λr.clone();
    V = V.reshape(size);
    return std::make_tuple(Λ, V);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
m00nny::singular_value_decomposition(torch::Tensor& X, int64_t lowrank, int64_t max_iters, double threshold)
{
    torch::NoGradGuard no_grad;
    // X: (*, n, m) where n >= m 
    int64_t dim = X.dim();
    int64_t n   = X.size(dim - 2);
    int64_t m   = X.size(dim - 1);

    int64_t lowrank_1 = (lowrank < 0) || (lowrank > m) ? m : lowrank;
    int64_t lowrank_2 = (lowrank < 0) || (lowrank > n) ? n : lowrank;

    torch::Tensor AAT = X.matmul(X.transpose(dim-2, dim-1)); // AAT : (*, n, n). AAT is the matrix to calculate the left singular vector matrix.
    torch::Tensor ATA = X.transpose(dim-2, dim-1).matmul(X); // ATA : (*, m, m). ATA is the matrix to calculate the right singular vector matrix.

    auto [Λ, U] = eigen_decomposition(AAT, lowrank_2, max_iters, threshold); // U : (*, n, n). U is the left singular vector matrix.
    auto [Σ, V] = eigen_decomposition(ATA, lowrank_1, max_iters, threshold); // V : (*, m, m). V is the right singular vector matrix.

    // SVD is the same as the eigen decomposition.
    return m > n ? std::make_tuple(U, Λ, V) : std::make_tuple(U, Σ, V);
}

PyObject*
linalg_kronecker_product
(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", "Y", NULL};
    PyObject *pyX, *pyY;
    torch::Tensor X, Y;
    
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", (char**)keywords, &pyX, &pyY)) return NULL;
    Py_XINCREF(pyX);
    Py_XINCREF(pyY);
    X = THPVariable_Unpack(pyX);
    Y = THPVariable_Unpack(pyY);
    ExitGILScope

    torch::Tensor _kronecker_product = m00nny::kronecker_product(X, Y);
    PyObject* py_kronecker_product;
    InitGILScope
    py_kronecker_product = THPVariable_Wrap(_kronecker_product);
    Py_XDECREF(pyX);
    Py_XDECREF(pyY);
    ExitGILScope
    return py_kronecker_product;
}

PyObject*
linalg_inverse
(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", NULL};
    PyObject* pyX;
    torch::Tensor X;
    
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)keywords, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    torch::Tensor _inv = m00nny::inverse(X);
    PyObject* py_inv;
    InitGILScope
    py_inv = THPVariable_Wrap(_inv);
    Py_XDECREF(pyX);
    ExitGILScope
    return py_inv;
}

PyObject*
linalg_orthogonalize
(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", NULL};
    PyObject* pyX;
    torch::Tensor X;
    
    InitGILScope
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", (char**)keywords, &pyX)) return NULL;
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    ExitGILScope

    torch::Tensor _inv = m00nny::orthogonalize(X);
    PyObject* py_inv;
    InitGILScope
    py_inv = THPVariable_Wrap(_inv);
    Py_XDECREF(pyX);
    ExitGILScope
    return py_inv;
}

PyObject* linalg_eigen_decomposition (PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", "lowrank", "max_iters", "threshold", NULL};
    PyObject *pyX, *Pyrank, *Pymax_iters, *Pythreshold;
    torch::Tensor X;
    int64_t lowrank = -1;
    int64_t max_iters = 1000;
    double threshold = 1e-8;
    
    InitGILScope
    int64_t _stack = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOOO", (char**)keywords, &pyX, &Pyrank, &Pymax_iters, &Pythreshold)) return NULL;
    IfSaveState(_stack, (pyX == nullptr) || Py_IsNone(pyX), [](){ PyErr_SetString(PyExc_TypeError, "X is required."); })
    IfSaveState(_stack, (Pyrank == nullptr) || Py_IsNone(Pyrank), [&Pyrank](){ Pyrank = PyLong_FromLong(-1); })
    IfSaveState(_stack, (Pymax_iters == nullptr) || Py_IsNone(Pymax_iters), [&Pymax_iters](){ Pymax_iters = PyLong_FromLong(1000); })
    IfSaveState(_stack, (Pythreshold == nullptr) || Py_IsNone(Pythreshold), [&Pythreshold](){ Pythreshold = PyFloat_FromDouble(1e-8); })
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    IfRestoreState(_stack, [&Pythreshold](){ Py_XDECREF(Pythreshold); })
    IfRestoreState(_stack, [&Pymax_iters](){ Py_XDECREF(Pymax_iters); })
    IfRestoreState(_stack, [&Pyrank](){ Py_XDECREF(Pyrank); })
    ExitGILScope

    auto [Λ, V] = m00nny::eigen_decomposition(X, lowrank, max_iters, threshold);
    PyObject* pyΛ, *pyV;
    InitGILScope
    pyΛ = THPVariable_Wrap(Λ);
    pyV = THPVariable_Wrap(V);
    Py_XDECREF(pyX);
    ExitGILScope
    return PyTuple_Pack(2, pyΛ, pyV);
}


PyObject* linalg_eigen_decomposition_new (PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", "lowrank", "max_iters", "threshold", NULL};
    PyObject *pyX, *Pyrank, *Pymax_iters, *Pythreshold;
    torch::Tensor X;
    int64_t lowrank = -1;
    int64_t max_iters = 1000;
    double threshold = 1e-8;
    
    InitGILScope
    int64_t _stack = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOOO", (char**)keywords, &pyX, &Pyrank, &Pymax_iters, &Pythreshold)) return NULL;
    IfSaveState(_stack, (pyX == nullptr) || Py_IsNone(pyX), [](){ PyErr_SetString(PyExc_TypeError, "X is required."); })
    IfSaveState(_stack, (Pyrank == nullptr) || Py_IsNone(Pyrank), [&Pyrank](){ Pyrank = PyLong_FromLong(-1); })
    IfSaveState(_stack, (Pymax_iters == nullptr) || Py_IsNone(Pymax_iters), [&Pymax_iters](){ Pymax_iters = PyLong_FromLong(1000); })
    IfSaveState(_stack, (Pythreshold == nullptr) || Py_IsNone(Pythreshold), [&Pythreshold](){ Pythreshold = PyFloat_FromDouble(1e-8); })
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    IfRestoreState(_stack, [&Pythreshold](){ Py_XDECREF(Pythreshold); })
    IfRestoreState(_stack, [&Pymax_iters](){ Py_XDECREF(Pymax_iters); })
    IfRestoreState(_stack, [&Pyrank](){ Py_XDECREF(Pyrank); })
    ExitGILScope

    auto [Λ, V] = m00nny::eigen_decomposition_new(X, lowrank, max_iters, threshold);
    PyObject* pyΛ, *pyV;
    InitGILScope
    pyΛ = THPVariable_Wrap(Λ);
    pyV = THPVariable_Wrap(V);
    Py_XDECREF(pyX);
    ExitGILScope
    return PyTuple_Pack(2, pyΛ, pyV);
}

PyObject* linalg_singular_value_decomposition (PyObject *self, PyObject *args, PyObject *kwds)
{
    const char* keywords[] = {"X", "lowrank", "max_iters", "threshold", NULL};
    PyObject *pyX, *Pyrank, *Pymax_iters, *Pythreshold;
    
    torch::Tensor X;
    int64_t lowrank = -1;
    int64_t max_iters = 1000;
    double threshold = 1e-8;

    InitGILScope
    int64_t _stack = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOOO", (char**)keywords, &pyX, &Pyrank, &Pymax_iters, &Pythreshold)) return NULL;
    IfSaveState(_stack, (pyX == nullptr) || Py_IsNone(pyX), [](){ PyErr_SetString(PyExc_TypeError, "X is required."); })
    IfSaveState(_stack, (Pyrank == nullptr) || Py_IsNone(Pyrank), [&Pyrank](){ Pyrank = PyLong_FromLong(-1); })
    IfSaveState(_stack, (Pymax_iters == nullptr) || Py_IsNone(Pymax_iters), [&Pymax_iters](){ Pymax_iters = PyLong_FromLong(1000); })
    IfSaveState(_stack, (Pythreshold == nullptr) || Py_IsNone(Pythreshold), [&Pythreshold](){ Pythreshold = PyFloat_FromDouble(1e-8); })
    Py_XINCREF(pyX);
    X = THPVariable_Unpack(pyX);
    IfRestoreState(_stack, [&Pythreshold](){ Py_XDECREF(Pythreshold); })
    IfRestoreState(_stack, [&Pymax_iters](){ Py_XDECREF(Pymax_iters); })
    IfRestoreState(_stack, [&Pyrank](){ Py_XDECREF(Pyrank); })
    ExitGILScope

    auto [U, Σ, V] = m00nny::singular_value_decomposition(X, lowrank, max_iters, threshold);
    PyObject* pyU, *pyΣ, *pyV;
    InitGILScope
    pyU = THPVariable_Wrap(U);
    pyΣ = THPVariable_Wrap(Σ);
    pyV = THPVariable_Wrap(V);
    Py_XDECREF(pyX);
    ExitGILScope
    return PyTuple_Pack(3, pyU, pyΣ, pyV);
}