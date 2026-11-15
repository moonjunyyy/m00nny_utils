"""RanPAC (McDonnell et al., NeurIPS 2023): random projections and
pre-trained models for continual learning.

Nothing is learned by gradient after the first task, so nothing is forgotten.
The class scores come from a ridge regression on randomly projected features,

    h = relu(f W_rand),   G += H^T H,   Q += H^T Y,   Wo = (G + ridge I)^-1 Q,

where f is the input of the wrapped module's head, W_rand is a frozen
Gaussian matrix (feature_dim, M), and G, Q are sums over every sample seen so
far: the result is the same in whatever order the tasks arrive.

* First task ("first-session adaptation"): whatever is trainable in the
  wrapped module trains through its own head, as usual.  At
  ``after_task(0)`` everything but that head is frozen for good.  With
  first_session=False the backbone is frozen from the start.
* ``after_task(task_id, loader=...)`` is required for every task: one pass
  over the loader (inputs and labels) updates G and Q and re-solves Wo.  The
  ridge is picked among 10^-8 .. 10^8 on a held-out fifth of that pass.
* In eval mode the handler returns the ridge scores once the current task
  has been through ``after_task``; before that, and in training mode, it
  returns the module's own head.  That head stays trainable only so that a
  generic training loop keeps running; from the second task on its epochs
  change nothing the ridge scores depend on (``frozen`` is then True, for a
  loop that wants to skip them).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._cl_handler import _CLHandler, all_reduce_

HEAD_NAMES = ('head', 'heads', 'fc', 'classifier')


class RanPACHandler(_CLHandler):
    def __init__(
        self,
        module: nn.Module,
        M: int = 10000,
        first_session: bool = True,
        num_classes: int = None,
        head_name: str = None,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(module)
        self.M = M                  # 0: no projection, ridge on f itself
        self.first_session = first_session
        self.seed = seed
        for _name in ((head_name,) if head_name else HEAD_NAMES):
            if isinstance(getattr(module, _name, None), nn.Module):
                self._head_name = _name
                break
        else:
            raise TypeError(
                f"{type(module).__name__} has no head among {HEAD_NAMES}; "
                "pass head_name=...")
        _head = getattr(module, self._head_name)
        self._original_head_forward = _head.forward
        if num_classes is None:
            _linears = [m for m in _head.modules() if isinstance(m, nn.Linear)]
            if not _linears:
                raise TypeError("pass num_classes=...: the head has no Linear")
            num_classes = _linears[-1].out_features
        self.num_classes = num_classes
        self.ridges = [10.0 ** _e for _e in range(-8, 9)]
        self.ridge = None
        self.frozen = False
        self._feature = None
        self._fitted = -1
        self.register_buffer("W_rand", None)
        self.register_buffer("G", None)
        self.register_buffer("Q", None)
        self.register_buffer("Wo", None)
        if not first_session:
            self._freeze()
        self.enable()

    # ── in / out: the head's forward also keeps its input ───────────────
    def enable(self):
        self._enabled = True
        getattr(self.module, self._head_name).forward = self._head_forward

    def disable(self):
        self._enabled = False
        getattr(self.module, self._head_name).forward = \
            self._original_head_forward

    def _head_forward(self, feature, *args, **kwargs):
        self._feature = feature
        return self._original_head_forward(feature, *args, **kwargs)

    def _freeze(self):
        for name, param in self.module.named_parameters():
            if not name.startswith(self._head_name + "."):
                param.requires_grad = False
        self.frozen = True

    def _project(self, feature):
        feature = feature.detach().float()
        if self.M <= 0:
            return feature
        if self.W_rand is None:
            _gen = torch.Generator().manual_seed(self.seed)
            self.W_rand = torch.randn(
                feature.shape[-1], self.M, generator=_gen).to(feature.device)
        return F.relu(feature @ self.W_rand)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        if not self._enabled or self.training or self.Wo is None \
                or self._fitted != self.task_id:
            return out
        return (self._project(self._feature) @ self.Wo.T).to(out.dtype)

    # ── statistics ──────────────────────────────────────────────────────
    def _choose_ridge(self, fit_g, fit_q, hold_h, hold_y):
        """Ridge with the least squared error on the held-out projections."""
        if hold_h.shape[0] == 0:
            return 1.0
        vals, vecs = torch.linalg.eigh(fit_g.double())
        vals = vals.clamp_min(0)
        _q = vecs.T @ fit_q.double()
        _h = hold_h.double() @ vecs
        losses = torch.stack([
            (_h @ (_q / (vals + _r).unsqueeze(1)) - hold_y.double())
            .pow(2).sum()
            for _r in self.ridges
        ])
        losses = torch.nan_to_num(losses, nan=float("inf"))
        return self.ridges[int(all_reduce_(losses).argmin())]

    @torch.no_grad()
    def _accumulate(self, loader):
        _was_training, _was_enabled = self.module.training, self._enabled
        self.module.eval()
        self.enable()
        _ref = next(self.module.parameters())
        fit_g = fit_q = hold_g = hold_q = None
        hold_h, hold_y = [], []
        _seen = 0
        for x, y in loader:
            x = x.to(device=_ref.device, dtype=_ref.dtype)
            y = y.to(device=_ref.device)
            self.module(x)
            h = self._project(self._feature)
            onehot = F.one_hot(y, self.num_classes).to(h.dtype)
            if fit_g is None:
                fit_g = h.new_zeros(h.shape[1], h.shape[1])
                fit_q = h.new_zeros(h.shape[1], self.num_classes)
                hold_g, hold_q = fit_g.clone(), fit_q.clone()
            # every fifth sample is held out for the choice of the ridge
            _index = torch.arange(_seen, _seen + y.shape[0], device=y.device)
            _hold = _index % 5 == 4
            _seen += y.shape[0]
            fit_g += h[~_hold].T @ h[~_hold]
            fit_q += h[~_hold].T @ onehot[~_hold]
            hold_g += h[_hold].T @ h[_hold]
            hold_q += h[_hold].T @ onehot[_hold]
            hold_h.append(h[_hold])
            hold_y.append(onehot[_hold])
        self.module.train(_was_training)
        if not _was_enabled:
            self.disable()
        if fit_g is None:
            raise RuntimeError("RanPAC: the loader gave no batch")
        all_reduce_(fit_g), all_reduce_(fit_q)
        all_reduce_(hold_g), all_reduce_(hold_q)
        self.ridge = self._choose_ridge(
            fit_g, fit_q, torch.cat(hold_h), torch.cat(hold_y))
        self.G = fit_g + hold_g if self.G is None else self.G + fit_g + hold_g
        self.Q = fit_q + hold_q if self.Q is None else self.Q + fit_q + hold_q
        _eye = torch.eye(self.G.shape[0], device=self.G.device,
                         dtype=torch.float64)
        self.Wo = torch.linalg.solve(
            self.G.double() + self.ridge * _eye, self.Q.double()).T.float()

    # ── the calls a training loop makes ─────────────────────────────────
    def after_task(self, task_id, classes=None, loader=None, **kwargs):
        if loader is None:
            raise RuntimeError(
                "RanPAC needs after_task(..., loader=...): its class "
                "statistics are built from inputs and labels")
        if not self.frozen:
            self._freeze()
        self._accumulate(loader)
        self._fitted = task_id
