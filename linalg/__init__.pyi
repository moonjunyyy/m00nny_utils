import torch
from typing import Tuple, TypeVar

Lambda = TypeVar('Lambda', torch.Tensor)
Sigma = TypeVar('Sigma', torch.Tensor)
V = TypeVar('V', torch.Tensor)
U = TypeVar('U', torch.Tensor)

def inverse(X: torch.Tensor) -> torch.Tensor: ...
def eigen_decomposition(X: torch.Tensor, lowrank: int = -1, max_iters:int=1000, threshold:float=1e-12) -> Tuple[Lambda, U]: ...
def singular_value_decomposition(X: torch.Tensor, lowrank: int = -1, max_iters:int=1000, threshold:float=1e-12) -> Tuple[U, Sigma, V]: ...