from . import utils

def _build() -> None:
    import os
    import sys
    import torch
    import subprocess
    from .utils.install_libtorch import install_libtorch
    from .utils.install_cuda_toolkit import install_cuda_toolkit

    # Get the system information
    _os_info = os.uname()
    _os_info = f"{_os_info.sysname}_{_os_info.release}_{_os_info.machine}" # For example, Linux_5.4.0-80-generic_x86_64

    # Get the python path for get the base path.
    _python_path = os.path.dirname(p=sys.executable)
    _base_env_path = os.path.dirname(p=_python_path)

    # Get the torch version and cuda version
    _torch_version = torch.__version__
    _torch_cuda_version = torch.version.cuda

    if not os.path.exists(path=os.path.join(_base_env_path, "bin", "gcc")):
        raise FileNotFoundError(f"GCC path not found: {_base_env_path}/bin/gcc.\n"
                                f"This module requires the GCC compiler to build the shared library.\n"
                                f"You may need to install the GCC compiler.")
    if not os.path.exists(path=os.path.join(_base_env_path, "bin", "g++")):
        raise FileNotFoundError(f"G++ path not found: {_base_env_path}/bin/g++.\n"
                                f"This module requires the G++ compiler to build the shared library.\n"
                                f"You may need to install the G++ compiler.")
    if not os.path.exists(path=os.path.join(_base_env_path, "bin", "cmake")):
        raise FileNotFoundError(f"CMake path not found: {_base_env_path}/bin/cmake.\n"
                                f"This module requires the CMake to build the shared library.\n"
                                f"You may need to install the CMake.")

    # For example, conda environment has the cuda toolkit installed.
    # But it has some files missing, we cannot perform the build.
    # So, we need to install the cuda toolkit.
    _cuda_base_path = f"{_base_env_path}/cuda-{_torch_cuda_version}"
    if not os.path.exists(path=_cuda_base_path):
        os.makedirs(name=_cuda_base_path)
        install_cuda_toolkit(version=_torch_cuda_version, installation_path=_cuda_base_path)

    _torch_cuda_version = _torch_cuda_version.replace(".", "") # For example, 102
    _libtorch_base_path = f"{_base_env_path}/libtorch_{_torch_version}_cu{_torch_cuda_version}"
    # Find the libtorch path, if not found, download the libtorch.
    if not os.path.exists(path=_libtorch_base_path):
        import zipfile
        import requests
        _libtorch_file_name = f"libtorch-cxx11-abi-shared-with-deps-{_torch_version}+cu{_torch_cuda_version}.zip"
        _libtorch_url = f"https://download.pytorch.org/libtorch/cu{_torch_cuda_version}/{_libtorch_file_name.replace('+', '%2B')}"
        res = requests.get(url=_libtorch_url)
        if res.status_code != 200:
            raise RuntimeError(f"Libtorch download failed with status code: {res.status_code}.\n"
                               f"Please check the libtorch url: {_libtorch_url}\n"
                               f"You may need to download the libtorch manually proper version from the link: {_libtorch_url}")
        with open(file=os.path.join(_base_env_path, _libtorch_file_name), mode="wb") as f:
            # Save the libtorch zip file
            f.write(res.content)
            # Extract the libtorch zip file to the base environment path (_base_env_path/libtorch)
            zipfile.ZipFile(file=f).extractall(path=_base_env_path)
        # Remove the libtorch zip file
        os.remove(path=os.path.join(_base_env_path, _libtorch_file_name))
        if not os.path.exists(path=os.path.join(_base_env_path,  "libtorch")):
            raise FileNotFoundError(f"Libtorch path not found: {_libtorch_base_path}.\n"
                                    f"You may need to download the libtorch manually proper version from the link: {_libtorch_url}")
        # Rename the path for version control
        os.rename(
            src=os.path.join(_base_env_path,  "libtorch"),
            dst=_libtorch_base_path
        )

    _build_id = f"{_os_info}_torch_{_torch_version}+cu{_torch_cuda_version}"
    _module_path = os.path.dirname(p=__file__)
    print(f"OS info: {_os_info}")
    print(f"Base environment path: {_base_env_path}")
    print(f"Build id: {_build_id}")
    print(f"Python path: {_python_path}")
    print(f"Python version: {sys.version}")
    print(f"CUDA root path: {_cuda_base_path}")
    print(f"CUDA version: {_torch_cuda_version}")
    print(f"Libtorch path: {_libtorch_base_path}")
    print(f"Libtorch version: {_torch_version}")
    print(f"Module path: {_module_path}")

    if not os.path.exists(path=os.path.join(_module_path, f"lib_{_build_id}")):
        _command = f"cd {_module_path} && "\
                   f"mkdir lib_{_build_id} && "\
                   f"mkdir objdir_{_build_id} && "\
                   f"cd objdir_{_build_id} && "\
                   f"cmake .. "\
                   f"-DPREFIX_PATH={_module_path}/lib_{_build_id} "\
                   f"-DCMAKE_C_COMPILER={_base_env_path}/bin/gcc "\
                   f"-DCMAKE_CXX_COMPILER={_base_env_path}/bin/g++ "\
                   f"-DCMAKE_CUDA_COMPILER={_cuda_base_path}/bin/nvcc "\
                   f"-DCMAKE_INCLUDE_PATH=\"{_base_env_path}/include;{_libtorch_base_path}/include;{_cuda_base_path}/include\" "\
                   f"-DCMAKE_LIBRARY_PATH=\"{_libtorch_base_path}/lib;{_cuda_base_path}/lib;{_libtorch_base_path}/lib64;{_cuda_base_path}/lib64;{_base_env_path}/lib;{_base_env_path}/lib64\""\
                   f"-DPYTHON_DEV_PATH={_base_env_path} "\
                   f"-DLIBTORCH_PATH={_libtorch_base_path}"\
                   f"-DCUDA_TOOLKIT_ROOT_DIR={_cuda_base_path} "\
                   f"-DCMAKE_BUILD_TYPE=Release "\
                   f"&& if [ $? -ne 0 ]; then rm -rf objdir_{_build_id} lib_{_build_id} && exit 1; fi "\
                   f"&& make -j$(nproc) "\
                   f"&& if [ $? -ne 0 ]; then rm -rf objdir_{_build_id} lib_{_build_id} && exit 1; fi "\
                   f"&& rm -rf objdir_{_build_id}"
    print(f"Building shared library with the command:\n{_command}")
    os.system(command=_command)
    if not os.path.exists(path=os.path.join(_module_path, f"lib_{_build_id}")):
        raise FileNotFoundError(f"Shared library build failed for the module: {_module_path} with build id: {_build_id}")
# Build the shared library
_build()