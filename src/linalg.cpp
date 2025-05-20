#include "linalg.h"


void
m00nny::internal::check_square(const torch::Tensor& X, const std::string& func_name) {
    if (X.dim() < 2) {
        throw std::runtime_error(func_name + ": Input tensor must have at least 2 dimensions. Got " + std::to_string(X.dim()));
    }
    int64_t n = X.size(-2);
    int64_t m = X.size(-1);
    if (n != m) {
        throw std::runtime_error(func_name + ": Input tensor's last two dimensions must be square. Got (" +
                                std::to_string(n) + ", " + std::to_string(m) + ")");
    }
}

// torch::Tensor
// m00nny::inverse(const torch::Tensor& X_const) {
//     LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::inverse\n");
//     m00nny::internal::check_square(X_const, "m00nny::inverse");

//     int64_t B = 1;
//     auto dims = X_const.dim();
//     const int64_t N = X_const.size(-1);
//     for (int64_t i = 0; i < dims - 2; ++i) { B *= X_const.size(i); }

//     const double epsilon = 1e-12;
//     auto options = X_const.options();
    
//     // Initial augmented matrix (B, N, 2N)
//     torch::Tensor A_aug = torch::cat({
//         X_const.clone().reshape({B, N, N}),
//         torch::eye(N, options).unsqueeze(0).expand({B, N, N}).clone()
//     }, 2);

//     // Row permutation buffer (B, N)
//     torch::Tensor row_perm = torch::arange(N, options.dtype(torch::kLong)).unsqueeze(0).expand({B, N}).clone();
//     torch::Tensor batch_idx = torch::arange(B, options.dtype(torch::kLong));
//     torch::Tensor all_k = torch::arange(N, options.dtype(torch::kLong));
//     torch::Tensor mask = torch::empty({B, N}, options.dtype(torch::kBool));
    
//     torch::Tensor long_buffer1 = torch::empty({B, N}, options.dtype(torch::kLong));
//     torch::Tensor long_buffer2 = torch::empty({B, N}, options.dtype(torch::kLong));
//     torch::Tensor float_buffer1 = torch::empty({B, N, 2 * N}, options);
//     torch::Tensor float_buffer2 = torch::empty({B, N, 2 * N}, options);
//     torch::Tensor float_buffer3 = torch::empty({B, N, 2 * N}, options);

//     for (int64_t i = 0; i < N; ++i) {
//         // === 1. Find pivot ===
//         torch::Tensor remaining_rows = row_perm.slice(1, i, N);              // (B, N-i)
//         torch::Tensor slice_rows = float_buffer1.slice(1, 0, N-i);
//         torch::gather_out(slice_rows, A_aug, 1, remaining_rows.unsqueeze(-1).expand({B, N - i, 2 * N}));
//         printf("4\n");
        
//         torch::Tensor col_i = float_buffer2.slice(1, 0, N-i).slice(2, 0, 1).select(2, 0);
//         col_i.copy_(slice_rows.select(2, i).abs());
//         printf("5\n");
        
//         auto [_, rel_pivot] = col_i.max(1);
//         torch::Tensor pivot_idx = rel_pivot + i;  // (B,)
//         printf("6\n");

//         // === 2. row_perm swap ===
//         torch::Tensor current = row_perm.index({batch_idx, torch::full({B}, i, row_perm.options().dtype(torch::kLong))});
//         torch::Tensor pivot = row_perm.index({batch_idx, pivot_idx});
//         row_perm.index_put_({batch_idx, i}, pivot);
//         row_perm.index_put_({batch_idx, pivot_idx}, current);
//         printf("7\n");

//         // === 3. Normalize pivot row ===
//         // 별도의 버퍼 사용
        
//         torch::Tensor pivot_row_indices = long_buffer2.slice(1, 0, 1);
//         pivot_row_indices.copy_(row_perm.index({batch_idx, i}));
//         printf("8\n");
        
//         torch::Tensor pivot_rows = A_aug.gather(1, pivot_row_indices.unsqueeze(-1).expand({B, 1, 2*N}));
//         torch::Tensor pivot_vals = pivot_rows.index({torch::indexing::Slice(), i});
//         if (pivot_vals.abs().lt(epsilon).any().item<bool>()) {
//             throw std::runtime_error("Singular matrix encountered.");
//         }
//         printf("9\n");

//         torch::Tensor pivot_rows_normalized = float_buffer3.slice(1, 0, 1);
//         torch::div_out(pivot_rows_normalized, pivot_rows, pivot_vals.unsqueeze(-1).expand({B, 1, 2*N}));
//         A_aug.scatter_(1, pivot_row_indices, pivot_rows_normalized.unsqueeze(1));
//         printf("10\n");
        
//         // === Remove pivot column from all rows (except i) ===
//         torch::not_equal_out(mask, all_k.unsqueeze(0).expand_as(mask), i);
//         printf("11\n");
        
//         // 별도의 버퍼 사용
//         torch::Tensor valid_k = all_k.masked_select(mask[0]);
//         torch::Tensor row_k_ids = row_perm.index({
//             batch_idx.unsqueeze(1), 
//             valid_k.unsqueeze(0).expand({B, N-1})
//         });

//         torch::Tensor gather_idx_k = row_k_ids.unsqueeze(-1).expand({B, N-1, 2*N});
//         torch::Tensor rows_k = float_buffer3.slice(1, 0, N-1);
//         rows_k.copy_(A_aug.gather(1, gather_idx_k));
//     }

//     // === 5. Reorder output rows ===
//     torch::Tensor final_indices = row_perm.unsqueeze(-1).expand({B, N, 2*N});
//     torch::Tensor reordered = A_aug.gather(1, final_indices);
//     return reordered.slice(2, N, 2*N).reshape_as(X_const);
// }

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

    // Step 2: Construct the C matrix. C_j = sum_{k=1}^{j-1} A_k
    torch::Tensor C = A.cumsum(/*dim=*/2) - A; // cumsum along cols_dim. 
    
    // Step 3: Modify and normalize columns of C to get C_prime (columns c'_j)
    if (m_cols > 0) C.select(/*dim=*/2, /*index=*/0).fill_(1.0); // Set first column of C to all ones.

    torch::Tensor col_norms_C = C.norm(2, /*dim=*/1, /*keepdim=*/true); // Norm along rows_dim
    C.div_(col_norms_C.clamp_min(epsilon));   // C now contains normalized c_j (let's call them c'_j)
                                            // c'_1 = normalize(vector_of_ones)
                                            // c'_2 = normalize(a_1) = a_1
                                            // c'_j = normalize(sum_{k=1}^{j-1} a_k) (where a_1 in sum was used after c_1=1 step's normalization if not careful)
                                            // This interaction of c_1=1 affecting normalization of subsequent sums is tricky.
                                            // It's cleaner if C was formed, then c_1 modified, then all columns normalized independently.
                                            // The current PyTorch norm will do this independently.
    // Step 4: Zero out the first column of normalized C
    if (m_cols > 0) C.select(/*dim=*/2, /*index=*/0).fill_(0.0); // So, effectively c'_1 (for projection) becomes 0.
    
    // Step 5: The core subtraction: A_out_j = A_j - <A_j, C'_j> * C'_j
    // Inner product <A_j, C'_j> means summing (A_ij * C'_ij) over i (rows)
    torch::Tensor inner_products = (A * C).sum(/*dim=*/1, /*keepdim=*/true); // Shape: (num_matrix, 1, m_cols)
                                                                             // Element (b, 0, j) is <A_b[:,j], C_b[:,j]>
    A.sub_(inner_products * C); // This is A_j_new = A_j_old - <A_j_old, C'_j> * C'_j
    // Step 6: Final normalization of resulting columns
    torch::Tensor final_col_norms = A.norm(2, /*dim=*/1, /*keepdim=*/true); // Norm along rows_dim
    A.div_(final_col_norms.clamp_min(epsilon));
    return A.reshape(size);
}

std::tuple<torch::Tensor, torch::Tensor>
m00nny::eigen_decomposition(const torch::Tensor& X, int64_t lowrank, int64_t max_iters, double threshold)
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

        _buffer_0_nrnr.copy_(dE_dV.permute({0, 1, 2, 3, 4}));         // (*, n, r, n, r)
        _buffer_1_nrrn.copy_(dE_dV.permute({0, 3, 4, 2, 1}));         // (*, n, r, r, n)
        torch::matmul_out(ddL1_dVdVt,
            _buffer_0.view({num_matrix, n * lowrank, lowrank * n}),
            _buffer_1.view({num_matrix, n * lowrank, lowrank * n}));   // ddL1_dVdVt : (*, n*r, r*n).
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
        torch::matmul_out(_buffer_1_like_vec_V, m00nny::inverse(ddL1_dVdVt), _buffer_0_like_vec_V); // _buffer_1 = ggL1_gVgVt^-1 gL1_gV
        V.sub_(_buffer_1_like_V); // V = V - _buffer_1

        _buffer_0_like_Vt.copy_(dL2_dV.transpose(1, 2)); // _buffer_0_like_Vt : (*, r, n).
        torch::matmul_out(_buffer_1_like_vec_V, m00nny::inverse(ddL2_dVdVt), _buffer_0_like_vec_V); // _buffer_2 = ggL2_gVgVt^-1 gL2_gV
        V.sub_(_buffer_1_like_V); // V = V - _buffer_1
    }
    V = V.reshape(size);
    return std::make_tuple(Λ, V);
}

std::tuple<torch::Tensor, torch::Tensor>
m00nny::eigen_decomposition_new(const torch::Tensor& X_const, int64_t lowrank, int64_t max_iters, double threshold, double damping_factor)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::eigen_decomposition\n")
     torch::NoGradGuard no_grad;
    internal::check_square(X_const, "m00nny::eigen_decomposition_newton_impl");
    // NOTE: We assume X_const is symmetric.

    int64_t num_matrix = X_const.numel() / (X_const.size(-2) * X_const.size(-1));
    int64_t n = X_const.size(-2);
    int64_t r = (lowrank <= 0 || lowrank > n) ? n : lowrank;

    if (r == 0) {
        return std::make_tuple(torch::empty({num_matrix, 0, 0}, X_const.options()),
                               torch::empty({num_matrix, n, 0}, X_const.options()));
    }

    torch::Tensor A = X_const.reshape({num_matrix, n, n});
    torch::Tensor V = torch::randn({num_matrix, n, r}, A.options());
    if (V.numel() > 0) V = m00nny::orthogonalize(V);

    torch::Tensor Vt, AV, Lambda_diag_r_mat, E_residual; // Renamed E to E_residual
    torch::Tensor Ir = torch::eye(r, r, A.options()).unsqueeze(0).expand({num_matrix, r, r});
    torch::Tensor In = torch::eye(n, n, A.options()).unsqueeze(0).expand({num_matrix, n, n}); // For n x n Identity
    const double epsilon_solve = 1e-9; // Tolerance for solve/inverse

    LIBM00NNY_DEBUG_MESSAGE("Starting Newton iterations for eigen_decomposition...\n");
    for (int64_t iter = 0; iter < max_iters; ++iter) {
        if (V.numel() == 0) break;

        if (iter > 0) { V = m00nny::orthogonalize(V); }
        Vt = V.transpose(1, 2).contiguous(); // (*, r, n)

        AV = torch::bmm(A, V); // (*, n, r)
        torch::Tensor vec_lambda = torch::einsum("bnr,bnr->br", {V, AV}); // (*, r) eigenvalues
        Lambda_diag_r_mat = torch::diag_embed(vec_lambda); // (*, r, r) diagonal matrix
        
        E_residual = AV - torch::bmm(V, Lambda_diag_r_mat); // (*, n, r)

        double fro_norm_E_sq = E_residual.pow(2).sum().item<double>(); // Sum of squares of all elements
        double avg_elem_E_sq = fro_norm_E_sq / std::max(1.0, static_cast<double>(E_residual.numel()));
        LIBM00NNY_DEBUG_MESSAGE("Iter " + std::to_string(iter) + ", Avg Elem |AV - VL|^2: " + std::to_string(avg_elem_E_sq) + "\n");
        if (std::sqrt(avg_elem_E_sq) < threshold) { LIBM00NNY_DEBUG_MESSAGE("Converged on residual.\n"); break; }

        // ========================================================================
        // CRITICAL SECTION: Implementing YOUR derived Newton update formulas
        // Using the structure from your original C++ code's variable names
        // ggL1_gVgVt -> H1_approx (n x n)
        // ggL2_gVgVt -> H2_approx (n x n)
        // gL1_gV -> grad_L_eig (n x r)
        // gL2_gV -> grad_L_ortho (n x r)
        // ========================================================================
        torch::Tensor grad_L_eig_update_step;
        torch::Tensor grad_L_ortho_update_step;

        // --- Loss 1: Eigenvalue Residual Component ---
        // Original gL1_gV = 2 * [(A - Λ_nn)^T E_residual - E_residual Λ_rr]
        // Where Λ_nn was VΛV^T and Λ_rr was Λ
        torch::Tensor VLambdaVt = torch::bmm(torch::bmm(V, Lambda_diag_r_mat), Vt); // (*, n, n)
        torch::Tensor A_minus_VLambdaVt = A - VLambdaVt; // (*, n, n)

        torch::Tensor term1_eig = torch::bmm(A_minus_VLambdaVt.transpose(1,2), E_residual); // (*, n, r)
        torch::Tensor term2_eig = torch::bmm(E_residual, Lambda_diag_r_mat); // (*, n, r)
        torch::Tensor gL1_gV = 2 * (term1_eig - term2_eig); // (*, n, r)

        // Original ggL1_gVgVt = 2 * [(A - 2VΛV^T)^T (A - 2VΛV^T) - 2 * (A - VΛV^T)VΛV^T]
        torch::Tensor A_minus_2VLambdaVt = A - 2 * VLambdaVt;
        torch::Tensor H1_term1 = torch::bmm(A_minus_2VLambdaVt.transpose(1,2), A_minus_2VLambdaVt); // (*, n, n)
        torch::Tensor H1_term2 = torch::bmm(A_minus_VLambdaVt, VLambdaVt); // (*, n, n)
        torch::Tensor H1_approx = 2 * (H1_term1 - 2 * H1_term2); // (*, n, n)

        try {
            // Add regularization to Hessian for stability if needed
            // H1_approx = H1_approx + torch::eye(n, A.options()).unsqueeze(0) * 1e-6;
            grad_L_eig_update_step = torch::linalg::solve(H1_approx, gL1_gV, true);
        } catch (const c10::Error& e) {
            LIBM00NNY_DEBUG_MESSAGE("Newton Eigen Step: linalg::solve for eig failed. Using gradient. Error: " + std::string(e.what()) + "\n");
            grad_L_eig_update_step = gL1_gV; // Fallback to gradient
        }


        // --- Loss 2: Orthogonality Component ---
        torch::Tensor O_mat = torch::bmm(Vt, V) - Ir; // (*, r, r)
        torch::Tensor gL2_gV = 4 * torch::bmm(V, O_mat);      // (*, n, r)

        // Original ggL2_gVgVt = 12 V V^T - 4 I_n
        torch::Tensor H2_approx = 12 * torch::bmm(V, Vt) - 4 * In; // (*, n, n)

        try {
            // Add regularization
            // H2_approx = H2_approx + torch::eye(n, A.options()).unsqueeze(0) * 1e-6;
            grad_L_ortho_update_step = torch::linalg::solve(H2_approx, gL2_gV, true);
        } catch (const c10::Error& e) {
            LIBM00NNY_DEBUG_MESSAGE("Newton Eigen Step: linalg::solve for ortho failed. Using gradient. Error: " + std::string(e.what()) + "\n");
            grad_L_ortho_update_step = gL2_gV; // Fallback to gradient
        }

        // --- Combine updates ---
        // These step_sizes scale the Newton steps dV = H_inv * g.
        // For true Newton, step_sizes should be 1.0. Use damping_factor for control.
        // If using fallback gradients, these are learning rates and should be small.
        bool eig_solve_failed = grad_L_eig_update_step.equal(gL1_gV); // Check if fallback was used
        bool ortho_solve_failed = grad_L_ortho_update_step.equal(gL2_gV);

        double actual_eig_step_scale = eig_solve_failed ? 0.0001 : 1.0; // Smaller if gradient descent
        double actual_ortho_step_scale = ortho_solve_failed ? 0.001 : 1.0;


        torch::Tensor dV = (actual_eig_step_scale * grad_L_eig_update_step +
                            actual_ortho_step_scale * grad_L_ortho_update_step);
        V.sub_(damping_factor * dV);
        // ========================================================================
        // End of CRITICAL Newton update section
        // ========================================================================

        if (iter == max_iters - 1) { LIBM00NNY_DEBUG_MESSAGE("Max iterations reached.\n"); }
    }

    if (V.numel() > 0) V = m00nny::orthogonalize(V);
    Vt = V.transpose(1, 2).contiguous();
    AV = torch::bmm(A, V);
    torch::Tensor final_eigenvalues_vec = torch::einsum("bnr,bnr->br", {V, AV});

    auto [sorted_Lambda_vals, sort_indices] = final_eigenvalues_vec.sort(/*dim=*/-1, /*descending=*/true);
    final_eigenvalues_vec = sorted_Lambda_vals;
    
    torch::Tensor V_output = V;
    if (V.numel() > 0 && sort_indices.numel() > 0 && r > 0) {
        torch::Tensor expanded_sort_indices = sort_indices.unsqueeze(1).expand({num_matrix, n, r});
        V_output = V.gather(/*dim=*/2, expanded_sort_indices);
    }
    
    c10::IntArrayRef original_X_sizes = X_const.sizes();
    std::vector<int64_t> V_output_shape_vec;
    for(size_t i=0; i < original_X_sizes.size() - 2; ++i) V_output_shape_vec.push_back(original_X_sizes[i]);
    V_output_shape_vec.push_back(n); 
    V_output_shape_vec.push_back(r); 
    
    return std::make_tuple(torch::diag_embed(final_eigenvalues_vec), V_output.reshape(V_output_shape_vec));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
m00nny::singular_value_decomposition(const torch::Tensor& X, int64_t lowrank, int64_t max_iters, double threshold, double damping_factor)
{
    LIBM00NNY_DEBUG_MESSAGE("C++: m00nny::singular_value_decomposition\n")
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
    return m > n ? std::make_tuple(U, Σ, V) : std::make_tuple(U, Λ, V);
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
    double threshold = 1e-6;
    
    InitGILScope
    int64_t _stack = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOOO", (char**)keywords, &pyX, &Pyrank, &Pymax_iters, &Pythreshold)) return NULL;
    IfSaveState(_stack, (pyX == nullptr) || Py_IsNone(pyX), [](){ PyErr_SetString(PyExc_TypeError, "X is required."); })
    IfSaveState(_stack, (Pyrank == nullptr) || Py_IsNone(Pyrank), [&Pyrank](){ Pyrank = PyLong_FromLong(-1); })
    IfSaveState(_stack, (Pymax_iters == nullptr) || Py_IsNone(Pymax_iters), [&Pymax_iters](){ Pymax_iters = PyLong_FromLong(1000); })
    IfSaveState(_stack, (Pythreshold == nullptr) || Py_IsNone(Pythreshold), [&Pythreshold](){ Pythreshold = PyFloat_FromDouble(1e-6); })
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
    double threshold = 1e-6;
    
    InitGILScope
    int64_t _stack = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOOO", (char**)keywords, &pyX, &Pyrank, &Pymax_iters, &Pythreshold)) return NULL;
    IfSaveState(_stack, (pyX == nullptr) || Py_IsNone(pyX), [](){ PyErr_SetString(PyExc_TypeError, "X is required."); })
    IfSaveState(_stack, (Pyrank == nullptr) || Py_IsNone(Pyrank), [&Pyrank](){ Pyrank = PyLong_FromLong(-1); })
    IfSaveState(_stack, (Pymax_iters == nullptr) || Py_IsNone(Pymax_iters), [&Pymax_iters](){ Pymax_iters = PyLong_FromLong(1000); })
    IfSaveState(_stack, (Pythreshold == nullptr) || Py_IsNone(Pythreshold), [&Pythreshold](){ Pythreshold = PyFloat_FromDouble(1e-6); })
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
    double threshold = 1e-6;

    InitGILScope
    int64_t _stack = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|$OOOO", (char**)keywords, &pyX, &Pyrank, &Pymax_iters, &Pythreshold)) return NULL;
    IfSaveState(_stack, (pyX == nullptr) || Py_IsNone(pyX), [](){ PyErr_SetString(PyExc_TypeError, "X is required."); })
    IfSaveState(_stack, (Pyrank == nullptr) || Py_IsNone(Pyrank), [&Pyrank](){ Pyrank = PyLong_FromLong(-1); })
    IfSaveState(_stack, (Pymax_iters == nullptr) || Py_IsNone(Pymax_iters), [&Pymax_iters](){ Pymax_iters = PyLong_FromLong(1000); })
    IfSaveState(_stack, (Pythreshold == nullptr) || Py_IsNone(Pythreshold), [&Pythreshold](){ Pythreshold = PyFloat_FromDouble(1e-6); })
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