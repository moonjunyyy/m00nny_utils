import torch
from typing import Optional, Literal, Union

class TSNE:
    """t-SNE with second-order steps: the direction solves (4 a L(P) + mu I) dY = -grad (the partial Hessian of
    the attractive term, a the exaggeration), the step length comes from a backtracking line search.
    'exact': dense P, O(N^2) per evaluation. 'barnes_hut': P on 3 * perplexity neighbours and Barnes-Hut
    repulsion (angle), O(N log N), n_components <= 3."""

    n_components: int
    perplexity: float
    learning_rate: float          # first trial step of the line search, in units of the second-order step
    early_exaggeration: float
    max_iter: int
    n_iter_without_progress: int
    batch_size: int
    init_method: Literal["pca", "random"]
    dist_method: Union[Literal["euclidean", "cosine", "manhattan"], str]
    tsne_method: Literal["exact", "barnes_hut"]
    angle: float
    random_state: int
    kl_divergence: float          # read-only, after fit_transform
    n_iter: int                   # read-only, after fit_transform

    def __init__(self,
                 n_components: int = 2,
                 *,
                 perplexity: Optional[float] = 30.0,
                 learning_rate: Optional[float] = 1.0,
                 early_exaggeration: Optional[float] = 12.0,
                 max_iter: Optional[int] = 1000,
                 n_iter_without_progress: Optional[int] = 300,
                 batch_size: Optional[int] = 1024,
                 init_method: Optional[Literal["random", "pca"]] = "pca",
                 dist_method: Optional[Union[Literal["euclidean", "cosine", "manhattan"], int]] = "euclidean",
                 tsne_method: Optional[Literal["exact", "barnes_hut"]] = "barnes_hut",
                 angle: Optional[float] = 0.5,
                 random_state: Optional[int] = None) -> None: ...
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor: ...
    def __call__(self, X: torch.Tensor) -> torch.Tensor: ...
