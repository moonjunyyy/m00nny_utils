"""CODA-Prompt (Smith et al., CVPR 2023) as a handler.

Each prompted layer holds, per task, a block of components P (prompt), K
(key) and A (attention).  A frozen-backbone query q weights the components,

    alpha_m = cos(q * A_m, K_m),      prompt = sum_m alpha_m P_m,

and the weighted prompt is split in halves and prepended to the keys and the
values of that layer's attention (prefix tuning).  Components of earlier
tasks stay in the sum but no longer train; a new task's block starts
orthogonal to everything already there.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._cl_handler import (
    _CLHandler, vit_dims, vit_head, vit_layers, vit_tokens,
)
from .prompt import _LayerPromptTuning


def _ortho_penalty(t: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(t.shape[0], device=t.device, dtype=t.dtype)
    return ((t @ t.T - eye) ** 2).mean()


class _LayerCodaPrompt(_LayerPromptTuning):
    def __init__(
        self,
        module: nn.Module,
        embed_dim: int = 768,
        num_heads: int = 12,
        embedding_key: str = 'cls',
        pool_size: int = 100,
        prompt_length: int = 8,
        num_tasks: int = 10,
        ortho_mu: float = 0.0,
        seed: int = 0,
        layer_index: int = 0,
        **kwargs,
    ):
        super().__init__(
            module=module,
            embed_dim=embed_dim,
            num_heads=num_heads,
            embedding_key=embedding_key,
            prompt_fn='prefix',
            pool_size=pool_size,
            prompt_length=prompt_length,
        )
        assert prompt_length % 2 == 0, 'prefix halves need an even length'
        self.per_task = pool_size // num_tasks
        assert self.per_task > 0, 'pool_size must be at least num_tasks'
        self.num_tasks = num_tasks
        self.ortho_mu = ortho_mu
        self.seed = seed
        self.layer_index = layer_index
        self._task_id = 0
        self._ortho = None

        # One block per task instead of the parent's single pool: a finished
        # block is frozen by requires_grad, so no optimizer touches it again
        # (weight decay included).
        del self.prompt_pool, self.prompt_key
        _m, _l, _d = self.per_task, prompt_length, embed_dim
        self.coda_p = nn.ParameterList(
            [nn.Parameter(torch.zeros(_m, _l, _d)) for _ in range(num_tasks)])
        self.coda_k = nn.ParameterList(
            [nn.Parameter(torch.zeros(_m, _d)) for _ in range(num_tasks)])
        self.coda_a = nn.ParameterList(
            [nn.Parameter(torch.zeros(_m, _d)) for _ in range(num_tasks)])

        self._out_proj = getattr(self.attn_module, 'out_proj', None)
        if self._out_proj is None:
            self._out_proj = self.attn_module.proj
        self._tuple_out = isinstance(self.attn_module, nn.MultiheadAttention)

        self._initialized = -1
        self.set_task(0)

    # ── task bookkeeping ────────────────────────────────────────────────
    def set_task(self, task_id: int):
        self._task_id = task_id
        for _t in range(self.num_tasks):
            for _list in (self.coda_p, self.coda_k, self.coda_a):
                _list[_t].requires_grad = (_t == task_id)
        while self._initialized < task_id:
            self._initialized += 1
            self._init_block(self._initialized)

    @torch.no_grad()
    def _init_block(self, task_id: int):
        """Orthonormal block, orthogonal to the span of the earlier blocks."""
        gen = torch.Generator().manual_seed(
            self.seed + 7919 * self.layer_index + task_id)
        for _list in (self.coda_p, self.coda_k, self.coda_a):
            new = _list[task_id]
            flat = torch.randn(
                new.shape[0], new[0].numel(),
                generator=gen, dtype=torch.float64)
            if task_id > 0:
                old = torch.cat([
                    p.detach().reshape(p.shape[0], -1)
                    for p in _list[:task_id]
                ]).double().cpu()
                q, _ = torch.linalg.qr(old.T)
                flat = flat - (flat @ q) @ q.T
            q, _ = torch.linalg.qr(flat.T)
            new.copy_(q.T.reshape(new.shape).to(new))

    def _blocks(self, _list):
        return torch.cat(list(_list[:self._task_id + 1]), dim=0)

    # ── forward ─────────────────────────────────────────────────────────
    def _prompt_selection(self, _input):
        K = self._blocks(self.coda_k)
        A = self._blocks(self.coda_a)
        P = self._blocks(self.coda_p)
        query = self.query_embed.to(K.dtype)
        attended = F.normalize(query.unsqueeze(1) * A.unsqueeze(0), dim=-1)
        alpha = (attended * F.normalize(K, dim=-1).unsqueeze(0)).sum(-1)
        prompt = torch.einsum('bm,mld->bld', alpha, P)
        if self.training and self.ortho_mu > 0:
            self._ortho = (
                _ortho_penalty(K) + _ortho_penalty(A)
                + _ortho_penalty(P.flatten(1))
            )
        return prompt.to(_input.dtype)

    def _prompt_forward(self, _input, *args, **kwargs):
        B, N, C = _input.shape
        q, k, v = self._get_qkv(_input)
        if hasattr(self.attn_module, 'q_norm'):
            q = self.attn_module.q_norm(q)
            k = self.attn_module.k_norm(k)
        if self.query_embed is not None:
            prompt = self._prompt_selection(_input)
            _half = self.prompt_length // 2
            pk = prompt[:, :_half].reshape(
                B, _half, self.num_heads, -1).permute(0, 2, 1, 3)
            pv = prompt[:, _half:].reshape(
                B, _half, self.num_heads, -1).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self._out_proj(x)
        if hasattr(self.attn_module, 'proj_drop'):
            x = self.attn_module.proj_drop(x)
        return (x, None) if self._tuple_out else x


class CodaPromptHandler(_CLHandler):
    """CODA-Prompt on a ViT (torchvision or timm).

    The backbone is frozen (heads excepted) unless freeze_backbone=False.
    ``aux_loss`` is ortho_mu times the orthogonality penalty of the
    components in use; it is zero when ortho_mu is 0.
    """

    def __init__(
        self,
        module: nn.Module,
        num_tasks: int = 10,
        num_heads: int = None,
        embed_dim: int = None,
        embedding_key: str = 'cls',
        prefix_tune_idx: List[int] = (0, 1, 2, 3, 4),
        pool_size: int = 100,
        prompt_length: int = 8,
        ortho_mu: float = 0.0,
        freeze_backbone: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(module)
        if embed_dim is None or num_heads is None:
            _dim, _heads = vit_dims(module)
            embed_dim = embed_dim or _dim
            num_heads = num_heads or _heads
        if freeze_backbone:
            self.freeze_backbone()
        self.ortho_mu = ortho_mu
        self._query_embed = None
        for _n, _layer in enumerate(vit_layers(module)):
            if _n in prefix_tune_idx:
                self.layers.append(_LayerCodaPrompt(
                    _layer,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    embedding_key=embedding_key,
                    pool_size=pool_size,
                    prompt_length=prompt_length,
                    num_tasks=num_tasks,
                    ortho_mu=ortho_mu,
                    seed=seed,
                    layer_index=_n,
                ))

    @property
    def prompts(self):
        return self.layers

    @property
    def query_embed(self):
        return self._query_embed

    @query_embed.setter
    def query_embed(self, value):
        self._query_embed = value
        for prompt in self.prompts:
            if value is None:
                prompt._query_embed = None
            else:
                prompt.query_embed = value

    def _aux(self):
        loss = self._zero()
        if self.training and self.ortho_mu > 0:
            for prompt in self.prompts:
                if prompt._ortho is not None:
                    loss = loss + prompt._ortho
        return self.ortho_mu * loss

    def forward_features(self, x):
        """Tokens of the prompted pass; the query comes from a plain pass."""
        if not self._enabled:
            return vit_tokens(self.module, x)
        self.disable()
        with torch.no_grad():
            query = vit_tokens(self.module, x)
        self.enable()
        self.query_embed = query
        tokens = vit_tokens(self.module, x)
        self.query_embed = None
        return tokens

    def forward(self, x):
        return vit_head(self.module, self.forward_features(x))
