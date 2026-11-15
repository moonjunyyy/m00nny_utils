import inspect
import importlib
from typing import Tuple

import torch


def get_dataset(
    name: str,
    path: str = "./data",
    download: bool = False,
    **kwargs
):
    pass


# Short names `name` may be given as, beyond the class names of torch.optim
# and torch.optim.lr_scheduler. A value is a class or its dotted import path;
# a project adds its own the same way.
OPTIMIZERS = {}
SCHEDULERS = {
    "cosine": f"{__package__}.lr_scheduler.warmup_cosine_anneling"
              ".WarmUpCosineAnnelingScheduler",
    "constant": f"{__package__}.lr_scheduler.warmup_constant"
                ".WarmUpConstantScheduler",
}


def _resolve(name, table, namespace, base, suffix=""):
    """The class `name` stands for, or None.

    `name` is a class, a key of `table`, the name of a subclass of `base` in
    `namespace` (case ignored, `suffix` optional: "step" is StepLR), or a
    dotted import path.
    """
    if isinstance(name, type):
        return name
    _key = name.lower()
    _found = table.get(_key)
    if _found is None:
        _classes = {
            _attr.lower(): _obj for _attr, _obj in vars(namespace).items()
            if isinstance(_obj, type) and issubclass(_obj, base)
            and _obj is not base and not _attr.startswith("_")
        }
        _found = _classes.get(_key, _classes.get(_key + suffix))
    if _found is None and "." in name:
        _found = name
    if isinstance(_found, str):
        _module, _, _attr = _found.rpartition(".")
        _found = getattr(importlib.import_module(_module), _attr)
    return _found


def _accepted(target, options):
    """The entries of `options` that `target` takes by name, its first
    (positional) argument aside."""
    _names = list(inspect.signature(target).parameters)[1:]
    return {_k: _v for _k, _v in options.items() if _k in _names}


# The defaults below are those of the "opt" and "lrs" groups of
# util/configbase.py: a call without a config builds what the config would.
def get_optimizer(
        parameters,
        *,
        name: str = "AdamW",
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.01,
        nesterov: bool = False,
        betas: Tuple[float, float] = (0.9, 0.999),
        **kwargs
):
    _optimizer = _resolve(
        name, OPTIMIZERS, torch.optim, torch.optim.Optimizer
    )
    if _optimizer is None:
        raise ValueError(f"Unsupported optimizer: {name}")
    # each optimizer gets the options it takes: Adam has no momentum,
    # RMSprop no dampening
    return _optimizer(
        parameters,
        **_accepted(_optimizer, dict(
            kwargs,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            betas=betas,
        ))
    )


def get_scheduler(
    target_optimizer,
    *,
    name: str = "constant",
    warmup_steps: int = 5,
    total_steps: int = 100,
    gamma: float = 0.9,
    step_size: int = 30,
    **kwargs
):
    _scheduler = _resolve(
        name, SCHEDULERS, torch.optim.lr_scheduler,
        torch.optim.lr_scheduler.LRScheduler, suffix="lr"
    )
    if _scheduler is None:
        raise ValueError(f"Unsupported scheduler: {name}")
    return _scheduler(
        target_optimizer,
        **_accepted(_scheduler, dict(
            kwargs,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            gamma=gamma,
            step_size=step_size,
        ))
    )
