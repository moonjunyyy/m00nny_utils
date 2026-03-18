import torch
from ..system.log import Log

log = Log(name="meta_device_context")


class MetaDeviceContext:
    def __enter__(self):
        self.original_register_parameter = torch.nn.Module.register_parameter

        def register_parameter_meta(module, name, param):
            if param is not None:
                param = torch.nn.Parameter(
                    param.to(device="meta"),
                    requires_grad=param.requires_grad,)

            self.original_register_parameter(module, name, param)
        torch.nn.Module.register_parameter = register_parameter_meta
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.nn.Module.register_parameter = self.original_register_parameter
