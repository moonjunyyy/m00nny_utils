import torch
from typing import Optional, Literal
class TSNE:
    def __init__(self,
                 n_components:int=2,
                 perplexity:Optional[float]=30.0,
                 learning_rate:Optional[float]=200.0,
                 early_exaggeration:Optional[float]=12.0,
                 max_iter:Optional[int]=1000,
                 n_iter_without_progress:Optional[int]=300,
                 batch_size:Optional[int]=1000,
                 init_method:Optional[Literal["random", "pca"]]="random",
                 dist_method:Optional[Literal["euclidean", "cosine", "manhattan"]]="euclidean",
                 tsne_method:Optional[Literal["exact", "barnes_hut"]]="exact",
                 angle:Optional[float]=0.5,
                 random_state:Optional[int]=None) -> None: ...
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor: ...