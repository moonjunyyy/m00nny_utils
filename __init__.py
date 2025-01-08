from . import utils

def _build() -> None:
    import os
    import sys
    import torch
    import subprocess
    from .utils import install_cuda_toolkit

    # Get the system information
    _os_info = os.uname()
    _os_info = f"{_os_info.sysname}_{_os_info.release}_{_os_info.machine}" # For example, Linux_5.4.0-80-generic_x86_64

    # Get the python path for get the base path.
    _python_path = os.path.dirname(p=sys.executable)
    _base_env_path = os.path.dirname(p=_python_path)

    # Get the torch version and cuda version
    _torch_version = torch.__version__
    _torch_cuda_version = torch.version.cuda
    _torch_cuda_version = _torch_cuda_version.replace(".", "") # For example, 102
    _libtorch_path = f"{_base_env_path}/libtorch"
    if not os.path.exists(path=_libtorch_path):
        _libtorch_file_name = f"libtorch-cxx11-abi-shared-with-deps-{_torch_version}+cu{_torch_cuda_version}.zip"
        _libtorch_url = f"https://download.pytorch.org/libtorch/cu{_torch_cuda_version}/{_libtorch_file_name.replace('+', '%2B')}"
        os.system(command=
                  f"cd {_base_env_path}"
                  f"&& wget {_libtorch_url}"
                  f"&& unzip {_libtorch_file_name}"
                  f"&& rm -rf {_libtorch_file_name}")
        if not os.path.exists(path=_libtorch_path): raise FileNotFoundError(f"Libtorch path not found: {_libtorch_path}.\n You may need to download the libtorch manually proper version from the link: {_libtorch_url}")
    _build_id = f"{_os_info}_torch_{_torch_version}+cu{_torch_cuda_version}"

    install_cuda_toolkit.install_cuda_toolkit(version=_torch_cuda_version, installation_path=_base_env_path)
    # Now, find for the nvcc and cuda library paths.
    _nvcc_path = subprocess.run(args=["which", "nvcc"], capture_output=True, text=True).stdout.strip()
    _cuda_root_path = os.path.dirname(p=os.path.dirname(p=_nvcc_path)) # Assuming that the nvcc is in the cuda/bin directory.

    _module_path = os.path.dirname(p=__file__)

    print(f"Building the shared library for the module: {_module_path} with build id: {_build_id}")
    print(f"Base environment path: {_base_env_path}")
    print(f"Libtorch path: {_libtorch_path}")
    print(f"CUDA root path: {_cuda_root_path}")
    print(f"Module path: {_module_path}")
    print(f"Build id: {_build_id}")
    print(f"OS info: {_os_info}")
    print(f"Python path: {_python_path}")
    print(f"NVCC path: {_nvcc_path}")
    print(f"CUDA root path: {_cuda_root_path}")
    print(f"Libtorch path: {_libtorch_path}")
    print(f"Libtorch version: {_torch_version}")
    print(f"CUDA version: {_torch_cuda_version}")

    if not os.path.exists(path=os.path.join(_module_path, f"lib_{_build_id}")):
        os.system(command=
                  f"cd {_module_path} && "
                  f"mkdir lib_{_build_id} && "
                  f"mkdir objdir_{_build_id} && "
                  f"cd objdir_{_build_id} && "
                  f"cmake -S .. "
                  f"-DCMAKE_C_COMPILER={_base_env_path}/bin/gcc "
                  f"-DCMAKE_CXX_COMPILER={_base_env_path}/bin/g++ " 
                  f"-DCMAKE_CUDA_COMPILER={_cuda_root_path}/bin/nvcc "
                  f"-DCMAKE_INCLUDE_PATH=\"{_base_env_path}/include;{_libtorch_path}/include\" "
                  f"-DPREFIX_PATH={_module_path}/lib_{_build_id} "
                  f"-DLIBTORCH_PATH={_libtorch_path} "
                  f"-DPYTHON_DEV_PATH={_base_env_path} "
                  f"-DCUDA_TOOLKIT_ROOT_DIR={_cuda_root_path} "
                  f"-DCMAKE_BUILD_TYPE=Release "
                  f"&& if [ $? -ne 0 ]; then rm -rf objdir_{_build_id} lib_{_build_id} && exit 1; fi "
                  f"&& make -j8"
                  f"&& if [ $? -ne 0 ]; then rm -rf objdir_{_build_id} lib_{_build_id} && exit 1; fi "
                  f"&& rm -rf objdir_{_build_id}")
    if not os.path.exists(path=os.path.join(_module_path, f"lib_{_build_id}")):
        raise FileNotFoundError(f"Shared library build failed for the module: {_module_path} with build id: {_build_id}")
# Build the shared library
_build()