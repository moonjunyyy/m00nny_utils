import argparse
import json
import os
import sys

# The flag that names settings files. Precedence: the defaults of the flags < the files, in the
# order given < the other flags of the command line.
IMPORT_FLAG = "--import"


def load_config(path):
    """A settings file as a nested dict: .json, .yaml / .yml (needs PyYAML), .toml."""
    _ext = os.path.splitext(path)[1].lower()
    if _ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    if _ext in (".yaml", ".yml"):
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    if _ext == ".toml":
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    raise ValueError(f"{path}: a .json, .yaml or .toml file is expected")


def dump_config(config, path):
    """The nested dict parse_args gives, as a .json file that --import reads back."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
        f.write("\n")


def flatten(config, prefix=()):
    """{"opt": {"lr": 1}} -> {"opt.lr": 1}: the keys are the dests of the flags."""
    _flat = {}
    for _key, _val in config.items():
        if isinstance(_val, dict):
            _flat.update(flatten(_val, prefix + (_key,)))
        else:
            _flat[".".join(prefix + (_key,))] = _val
    return _flat


def nest(flat):
    """{"opt.lr": 1} -> {"opt": {"lr": 1}}."""
    _ret = {}
    for _dest, _val in flat.items():
        _cur = _ret
        *_tree, _leaf = _dest.split(".")
        for _it in _tree:
            _cur = _cur.setdefault(_it, {})
        _cur[_leaf] = _val
    return _ret


class ConfigBase:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            IMPORT_FLAG, dest="import", nargs="+", action="extend", default=[],
            metavar="FILE",
            help="Settings files (.json, .yaml, .toml), nested as the groups "
            "are: {\"epochs\": 20, \"opt\": {\"lr\": 1e-4}}. A later file "
            "overrides an earlier one, and the other flags override the files.")
        self._config_groups = {}

        default_group = self.add_group("Default", prefix="")
        default_group.add_argument(
            "--mode", type=str, default="train",
            help="Mode to run the script in (e.g., train, eval, test)")
        default_group.add_argument(
            "--epochs", type=int, default=100,
            help="Number of epochs to train for")
        default_group.add_argument(
            "--batchsize", type=int, default=64,
            help="Batch size for training")
        default_group.add_argument(
            "--exp-id", type=str, default="default_exp",
            help="Experiment ID for logging and checkpointing")
        default_group.add_argument(
            "--device", type=str, default="cuda",
            help="Device to use for training (e.g., cuda, cpu)")
        default_group.add_argument(
            "--dtype", type=str, default="float32",
            help="Data type to use for training"
            "(e.g., float32, float16, bfloat16)")
        default_group.add_argument(
            "--random-seed", type=int, default=None,
            help="Random seed for reproducibility (default: None)")
        default_group.add_argument(
            "--num-workers", type=int, default=4,
            help="Number of worker threads for data loading")
        default_group.add_argument(
            "--save-path", type=str, default="./output",
            help="Directory to save checkpoints and logs")

        distributed_group = self.add_group(
            "Distributed Training", prefix="dist")
        distributed_group.add_argument(
            "--global-world-size", type=int, default=1,
            help="Number of distributed processes for training")
        distributed_group.add_argument(
            "--global-rank", type=int, default=0,
            help="The begining ordinal of global rank for current machine")
        distributed_group.add_argument(
            "--local-world-size", type=int, default=1,
            help="Number of devices on the current machine.")
        distributed_group.add_argument(
            "--local-rank", type=int, default=0,
            help="The ordinal of local rank for current device.\n"
            "(This argument is placeholder, and the input will be ignored.)")
        distributed_group.add_argument(
            "--backend", type=str, default="nccl",
            help="Distributed backend to use (e.g., nccl, gloo, mpi)")
        distributed_group.add_argument(
            "--protocol", type=str, default="tcp://",
            help="URL for distributed training initialization")
        distributed_group.add_argument(
            "--master-addr", type=str, default="localhost",
            help="Master node address for distributed training")
        distributed_group.add_argument(
            "--master-port", type=str, default="42355",
            help="Master node port for distributed training")

        tp_group = self.add_group("Tensor Parallelism", prefix="tp")
        tp_group.add_argument(
            "--tp-width", type=int, default=1,
            help="Tensor parallelism width for distributed training")
        tp_group.add_argument(
            "--row-tp", type=str, nargs="*", default=[],
            help="Regex expression to apply row-wise tensor parallelism")
        tp_group.add_argument(
            "--col-tp", type=str, nargs="*", default=[],
            help="Regex expression to apply column-wise tensor parallelism")
        tp_group.add_argument(
            "--seq-tp", type=str, nargs="*", default=[],
            help="Regex expression to apply sequence-wise tensor parallelism")
        tp_group.add_argument(
            "--conv-tp", type=str, nargs="*", default=[],
            help="Regex expression to apply parallelism on convolutional layers")

        optim_group = self.add_group("Optimizers", "opt")
        optim_group.add_argument(
            "--name", type=str, default="AdamW",
            help="Optimizer to use for training "
            "(e.g., AdamW, Adam, SGD, RMSprop)")
        optim_group.add_argument(
            "--lr", type=float, default=1e-3,
            help="Learning rate for the optimizer")
        optim_group.add_argument(
            "--betas", type=float, nargs=2, default=(0.9, 0.999),
            help="Beta coefficients for AdamW optimizer (beta1 beta2)")
        optim_group.add_argument(
            "--momentum", type=float, default=0.9,
            help="Momentum for optimizers that support it (e.g., SGD)")
        optim_group.add_argument(
            "--dampening", type=float, default=0.0,
            help="Dampening for momentum (SGD)")
        optim_group.add_argument(
            "--weight-decay", type=float, default=0.01,
            help="Weight decay for the optimizer")
        optim_group.add_argument(
            "--nesterov", action="store_true", dest="nesterov", default=False,
            help="Whether to use Nesterov momentum (SGD)")
        optim_group.add_argument(
            "--no-init-every-task", action="store_false",
            dest="init_every_task", default=True,
            help="Whether to reinitialize the optimizer at the beginning of each task"
        )

        scheduler_group = self.add_group("LR Scheduler", "lrs")
        scheduler_group.add_argument(
            "--name", type=str, default="constant",
            help="Learning rate scheduler to use for Training "
            "(e.g., cosine, constant, exponential, step)")
        scheduler_group.add_argument(
            "--warmup-steps", type=int, default=5,
            help="Number of warmup steps for learning rate scheduler")
        scheduler_group.add_argument(
            "--total-steps", type=int, default=100,
            help="Total number of training steps for learning rate scheduler")
        scheduler_group.add_argument(
            "--gamma", type=float, default=0.9,
            help="Decay factor for exponential learning rate scheduler")
        scheduler_group.add_argument(
            "--step-size", type=int, default=30,
            help="Step size for step decay learning rate scheduler")

        dataset_group = self.add_group("Datasets", "data")
        dataset_group.add_argument(
            "--name", type=str, default="cifar100",
            help="Dataset to use for training "
            "(e.g., cifar100, tinyimagenet, imagenet_r)")
        dataset_group.add_argument(
            "--path", type=str, default="./data",
            help="Path to the dataset")
        dataset_group.add_argument(
            "--no-download", action="store_false",
            dest="download", default=True,
            help="Whether to download the dataset if not found")

    def add_group(self, group_name, prefix=""):
        self._config_groups[group_name] = _Config_Group(
            self.parser, group_name, prefix=prefix
        )
        return self._config_groups[group_name]

    def get_group(self, group_name):
        return self._config_groups.get(group_name, None)

    def del_group(self, group_name):
        _group = self._config_groups.pop(group_name, None)
        if _group is not None:
            _group._drop()

    def parse_args(self, args=None):
        _argv = sys.argv[1:] if args is None else list(args)
        # the settings files first: what they hold becomes the defaults the
        # command line is parsed against
        _pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        _pre.add_argument(
            IMPORT_FLAG, dest="files", nargs="+", action="extend", default=[])
        _files = _pre.parse_known_args(_argv)[0].files
        _actions = {_a.dest: _a for _a in self.parser._actions}
        for _file in _files:
            for _dest, _val in flatten(load_config(_file)).items():
                _action = _actions.get(_dest)
                if _action is None or _dest == "import":
                    self.parser.error(f"{_file}: unknown setting '{_dest}'")
                if _action.choices is not None and _val not in _action.choices:
                    self.parser.error(
                        f"{_file}: '{_dest}' is {_val!r}, not one of "
                        f"{list(_action.choices)}")
                self.parser.set_defaults(**{_dest: _val})
        parsed = vars(self.parser.parse_args(_argv))
        parsed.pop("import", None)
        return nest(parsed)


class _Config_Group:
    """The flags of one group. With the prefix `opt`, add_argument("--lr") is
    the flag --opt-lr and the setting opt.lr; a group of a group (add_group)
    goes one level further: --opt-sched-warmup, opt.sched.warmup."""

    def __init__(self, parser, name, prefix="", parent=None):
        self.name = name
        self.prefix = prefix.lstrip("-").replace(" ", "_").lower()
        self.path = tuple(
            _p for _p in (parent.path if parent else ()) + (self.prefix,) if _p)
        self.parser = parser
        self.group = self.parser.add_argument_group(
            name, f"(--{'-'.join(self.path)}-...)" if self.path else None
        )
        self.groups = {}

    def add_group(self, group_name, prefix):
        self.groups[group_name] = _Config_Group(
            self.parser, group_name, prefix=prefix, parent=self
        )
        return self.groups[group_name]

    def add_argument(self, *option_strings, **kwargs):
        if len(option_strings) == 1 and not isinstance(option_strings[0], str):
            option_strings = tuple(option_strings[0])
        _action = kwargs.get("action", None)
        _name = option_strings[0].lstrip("-")
        # an explicit dest is kept, under the path of the group
        _dest = kwargs.pop("dest", None)
        if _dest is None:
            _dest = _name.replace("-", "_")
            # "--no-x" of a store_false flag stores into "x"
            if _action == "store_false" and _dest.startswith("no_"):
                _dest = _dest[len("no_"):]
        _takes_value = _action in (None, "store", "append", "extend")
        if _takes_value and kwargs.get("metavar") is None:
            kwargs["metavar"] = _name.upper().replace("-", "_")
        return self.group.add_argument(
            *["--" + "-".join(self.path + (_o.lstrip("-"),))
              for _o in option_strings],
            dest=".".join(self.path + (_dest,)),
            **kwargs
        )

    def del_argument(self, dest):
        """Remove a setting of the group, by its name in the group ("lr", or the flag "--lr")."""
        _dest = ".".join(self.path + (dest.lstrip("-").replace("-", "_"),))
        for _action in [
            action for action in self.group._group_actions
            if action.dest == _dest
        ]:
            self._remove(_action)

    def _drop(self):
        # the group, its flags and the groups under it, out of the parser
        for _sub in self.groups.values():
            _sub._drop()
        self.groups = {}
        for _action in list(self.group._group_actions):
            self._remove(_action)
        self.parser._action_groups.remove(self.group)

    def _remove(self, action):
        # off the parser too: its options and its default go with it
        self.group._remove_action(action)
        for _option in action.option_strings:
            self.parser._option_string_actions.pop(_option, None)
        self.parser._defaults.pop(action.dest, None)
