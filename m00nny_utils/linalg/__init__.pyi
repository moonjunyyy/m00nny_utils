import torch
from typing import Tuple

def inverse(X: torch.Tensor) -> torch.Tensor:
    """Inverse of square matrices X (*, n, n), Gauss-Jordan with partial pivoting."""
    ...

def orthogonalize(X: torch.Tensor) -> torch.Tensor:
    """Orthonormal columns of X (*, n, m), in order, by Gram-Schmidt: each fixed basis' subtraction term is
    accumulated (cumulative sum) into every later column. Columns are taken as independent."""
    ...

def eigen_decomposition(X: torch.Tensor, *, lowrank: int = -1, max_iters: int = 1000,
                        threshold: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric X (*, n, n) = V L V^T: second-order (Newton, trust region) minimization of |X - V L V^T|^2 over
    orthonormal V with L the Rayleigh quotients, which selects the pairs of largest |eigenvalue|.

    Returns L (*, r, r) diagonal with the eigenvalues descending and V (*, n, r); r = lowrank (<= 0: n). With
    r < n the subspace is selected on X^2 first (no sign barrier), then finished on X. A pair is converged when
    |Xv - lv| <= threshold * |X|_F.
    """
    ...

def singular_value_decomposition(X: torch.Tensor, *, lowrank: int = -1, max_iters: int = 1000,
                                 threshold: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin X (*, n, m) = U S V^T from the eigen decomposition of the smaller Gram matrix.

    Returns U (*, n, k), S (*, k, k) diagonal descending, V (*, m, k); k = min(lowrank, n, m) (lowrank <= 0: all).
    """
    ...
