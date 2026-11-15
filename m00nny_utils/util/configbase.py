import argparse


class ConfigBase:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
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
        if group_name in self._config_groups:
            del self._config_groups[group_name]

    def parse_args(self):
        parsed = vars(self.parser.parse_args())
        ret = {}
        for _dest, _val in parsed.items():
            _cur = ret
            _dest_tree = _dest.split(".")
            for _it in _dest_tree[:-1]:
                if _it not in _cur:
                    _cur[_it] = {}
                _cur = _cur[_it]
            _cur[_dest_tree[-1]] = _val
        return ret


class _Config_Group_Action_Base(argparse.Action):
    def __init__(
        self,
        option_strings,
        **kwargs
    ):
        _option_strings = [
            option.replace(".", "-") for option in option_strings
        ]
        super().__init__(
            option_strings=_option_strings,
            **kwargs
        )
        _first_option_str = option_strings[0].lstrip("-")
        self._dest_group = (
            _first_option_str.split(".")[0]
            if "." in _first_option_str else None
        )

    def __call__(
        self,
        parser,
        namespace,
        values,
        option_string=None
    ):
        setattr(namespace, self.dest, values)


class _Config_Group:
    def __init__(self, parser, name, prefix=""):
        self.name = name
        self.prefix = prefix.lstrip("-").replace(" ", "_").lower()
        self.parser = parser
        self.group = self.parser.add_argument_group(
            f"{name}" if prefix else name,
            f"(--{prefix}-...)"
        )

    def add_argument(
        self,
        option_strings,
        **kwargs
    ):
        if isinstance(option_strings, str):
            option_strings = [option_strings]
        _dest = None
        _metavar = kwargs.pop("metavar", None)
        _action = kwargs.get("action", None)
        _option_strings = []
        _prefix = f"--{self.prefix}." if self.prefix else "--"
        for _option in option_strings:
            _flag = _option.lstrip("-")
            if _dest is None:
                _dest = f"{self.prefix}.{_flag.replace('-', '_')}"
                _dest = _dest.lstrip(".")
            if _action in ["store_true", "store_false"]:
                if _metavar is None:
                    _metavar = ""
                if _dest is not None:
                    _dest = _dest.replace("no_", "")
                kwargs["nargs"] = 0
                kwargs["const"] = True if _action == "store_true" else False
                kwargs["default"] = False if _action == "store_true" else True

            if _metavar is None:
                _metavar = _flag.upper().replace("-", "_")
            _option = _prefix + _flag.replace("_", "-")
            _option_strings.append(_option)
        kwargs["metavar"] = _metavar
        kwargs["dest"] = _dest
        kwargs["action"] = _Config_Group_Action_Base
        self.group.add_argument(
            *_option_strings,
            **kwargs
        )

    def del_argument(self, dest):
        _dest = f"{self.prefix}.{dest}" if self.prefix else dest
        self.group._group_actions = [
            action for action in self.group._group_actions
            if action.dest != _dest
        ]


if __name__ == "__main__":
    config = ConfigBase()
    args = config.parse_args()
    print(args)
