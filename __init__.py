from . import algorithm
from . import linalg
from .system.log import Log as Log

def __bootstrap__() -> None:
    import multiprocessing as mp

    import os
    from .system.log import Log
    _log_level = os.environ.get("LIBM00NNY_LOG_LEVEL", "INFO")
    _log_path  = os.environ.get("LIBM00NNY_LOG_PATH", None)
    if _log_path: _log_path = os.path.abspath(path=_log_path)
    log = Log(name="m00nny_utils_init", global_level=_log_level, path=_log_path)

    import sys
    import json
    import torch
    import shutil
    from .util.version import Version
    from .util.integrity_verify import integrity_verify_dir, integrity_sign_dir
    from .install.install_requirements import install_cuda_libs, install_libtorch, install_nvtx

    __lock = mp.Lock()
    FORCE_BUILD: int    = int(os.environ.get("LIBM00NNY_FORCE_REBUILD",  "0"))
    ROOT_DIR: str       = os.path.dirname(p=os.path.dirname(p=sys.executable))
    CURRENT_DIR: str    = os.path.dirname(p=__file__)
    
    _os_info = os.uname()    
    OS: str             = _os_info.sysname.lower()
    OS_VERSION: str     = _os_info.release
    OS_ARCH: str        = _os_info.machine.lower()
    
    _torch_version = torch.__version__
    _cuda_version = torch.version.cuda
    _torch_cudnn_version = torch.backends.cudnn.version() # 90100
    USE_CXX11_ABI: bool = torch._C._GLIBCXX_USE_CXX11_ABI # True or False
    
    PYTHON_PATH: str        = sys.executable
    PYTHON_VERSION: Version = Version(sys.version)
    TORCH_VERSION: Version  = Version(_torch_version)
    CUDA_VERSION: Version   = Version(_cuda_version)
    CUDNN_VERSION: Version  = Version(_torch_cudnn_version // 10000, _torch_cudnn_version % 10000 // 100, _torch_cudnn_version % 100)
    
    USE_NVTX: int        = int(os.environ.get("LIBM00NNY_USE_NVTX",       "1"))
    USE_CUDNN: int       = int(os.environ.get("LIBM00NNY_USE_CUDNN",      "1"))
    USE_CUDSS: int       = int(os.environ.get("LIBM00NNY_USE_CUDSS",      "1"))
    USE_CUSPARSELT: int  = int(os.environ.get("LIBM00NNY_USE_CUSPARSELT", "1"))

    DEBUG: int           = int(os.environ.get("LIBM00NNY_DEBUG",          "0"))

    # Check the GCC, G++, CMake
    assert os.path.isfile(path=os.path.join(ROOT_DIR, "bin", "gcc")), "GCC path not found."
    assert os.path.isfile(path=os.path.join(ROOT_DIR, "bin", "g++")), "G++ path not found."
    assert os.path.isfile(path=os.path.join(ROOT_DIR, "bin", "cmake")), "CMake path not found."

    with __lock:
        # Check the CUDA Toolkit
        CUDA_PATH = install_cuda_libs(lib_name="cuda", os_name=OS, os_arch=OS_ARCH, root_dir=ROOT_DIR, version=CUDA_VERSION, cuda_version=CUDA_VERSION, condition=lambda kv: kv[1]['license'] == "CUDA Toolkit")
        assert os.path.isfile(path=os.path.join(CUDA_PATH, "bin", "nvcc")), "Nvcc path not found."
        _cmake_paths = [CUDA_PATH]
        # Check the Libtorch
        LIBTORCH_PATH = install_libtorch(os_name=OS, root_dir=ROOT_DIR, version=TORCH_VERSION, cuda_version=CUDA_VERSION, use_cxx11_abi=USE_CXX11_ABI)
        assert os.path.isfile(path=os.path.join(LIBTORCH_PATH, "lib", "libtorch.so")), "Libtorch shared library path not found."
        _cmake_paths.append(LIBTORCH_PATH)

        if USE_CUDNN:
            # Check the CuDNN
            CUDNN_PATH = install_cuda_libs(lib_name="cudnn", os_name=OS, os_arch=OS_ARCH, root_dir=ROOT_DIR, version=CUDNN_VERSION, cuda_version=CUDA_VERSION, condition=lambda kv: True)
            assert os.path.isfile(path=os.path.join(CUDNN_PATH, "lib", "libcudnn.so")), "CuDNN shared library path not found."
            _cmake_paths.append(CUDNN_PATH)
        if USE_CUDSS:
            # Check the CuDSS
            CUDSS_PATH = install_cuda_libs(lib_name="cudss", os_name=OS, os_arch=OS_ARCH, root_dir=ROOT_DIR, version=None, cuda_version=CUDA_VERSION, condition=lambda kv: True)
            assert os.path.isfile(path=os.path.join(CUDSS_PATH, "lib", "libcudss.so")), "CuDSS shared library path not found."
            _cmake_paths.append(CUDSS_PATH)
        if USE_CUSPARSELT:
            # Check the CuSparseLT
            CUSPARSELT_PATH = install_cuda_libs(lib_name="cusparselt", os_name=OS, os_arch=OS_ARCH, root_dir=ROOT_DIR, version=None, cuda_version=CUDA_VERSION, condition=lambda kv: True)
            assert os.path.isfile(path=os.path.join(CUSPARSELT_PATH, "lib", "libcusparseLt.so")), "CuSparseLT shared library path not found."
            _cmake_paths.append(CUSPARSELT_PATH)
        # Check the NVTX
        if USE_NVTX:
            _ = install_nvtx(root_dir=CURRENT_DIR)
        try:
            if FORCE_BUILD: log.info("Forcing to build the shared library..."); raise RuntimeError("Forcing to build the shared library...")
            _lib_installation_path = os.path.join(CURRENT_DIR, "debug" if DEBUG else "release")
            if not os.path.isdir(s=_lib_installation_path): os.makedirs(name=_lib_installation_path); raise RuntimeError("Shared library is not found.")
            if os.path.isdir(s=_lib_installation_path) and os.path.isdir(s=os.path.join(_lib_installation_path, "lib")):
                with open(file=os.path.join(_lib_installation_path, "build_info.json"), mode="r") as f:
                    _info = json.load(fp=f)
                if not (\
                _info["OS"]         == f"{OS}-{OS_ARCH}" and\
                _info["Python"]     == str(PYTHON_VERSION) and\
                _info["CUDA"]       == str(CUDA_VERSION) and\
                _info["Libtorch"]   == str(TORCH_VERSION) and\
                _info["CUDNN"]      == str(CUDNN_VERSION) if USE_CUDNN else None and\
                _info["CUDSS"]      == USE_CUDSS if USE_CUDSS else None and\
                _info["CUSPARSELT"] == USE_CUSPARSELT if USE_CUSPARSELT else None and\
                _info["NVTX"]       == USE_NVTX if USE_NVTX else None):
                    raise RuntimeError("Shared library is not up-to-date.")
                
            if not (\
                integrity_verify_dir(directory_path=os.path.join(CURRENT_DIR, "src"),     
                                     hash_file_path=os.path.join(CURRENT_DIR, 'debug' if DEBUG else 'release', '.src.sha256')) and\
                integrity_verify_dir(directory_path=os.path.join(CURRENT_DIR, "include"), 
                                     hash_file_path=os.path.join(CURRENT_DIR, 'debug' if DEBUG else 'release', '.include.sha256'))):
                raise RuntimeError("Source code is changed.")
            log.info("Shared library is up-to-date.\n")
        except Exception as e:
            log.info("Source code is changed or library is not found. Building the shared library...\n")
            log.debug(f"{'OS info:':<20}{OS}-{OS_VERSION}-{OS_ARCH}")
            log.debug(f"{'Root path:':<20}{ROOT_DIR}")
            log.debug(f"{'Python version:':<20}{PYTHON_VERSION}")
            log.debug(f"{'Python path:':<20}{PYTHON_PATH}")
            log.debug(f"{'CUDA version:':<20}{CUDA_VERSION}")
            log.debug(f"{'CUDA path:':<20}{CUDA_PATH}")
            log.debug(f"{'Libtorch version:':<20}{TORCH_VERSION}")
            log.debug(f"{'Libtorch path:':<20}{LIBTORCH_PATH}")
            log.debug(f"{'CUDNN info:':<20}{CUDNN_VERSION if USE_CUDNN else None}")
            log.debug(f"{'CUDSS info:':<20}{USE_CUDSS if USE_CUDSS else None}")
            log.debug(f"{'CUSPARSELT info:':<20}{USE_CUSPARSELT if USE_CUSPARSELT else None}")
            log.debug(f"{'NVTX info:':<20}{USE_NVTX if USE_NVTX else None}")
            log.debug(f"{'Debug mode:':<20}{'Debug' if DEBUG else 'Release'}")
            log.debug(f"{'Module path:':<20}{CURRENT_DIR}")
            log.debug(f"CMAKE paths:\n\t{'\n\t'.join(_cmake_paths)}")
            _command = f'''
#!/bin/bash

# AUTO GENERATED BUILD SCRIPT
# DO NOT MODIFY THIS SCRIPT

cd {CURRENT_DIR}
mkdir objdir
cd objdir
cmake .. \\
-DPREFIX_PATH={CURRENT_DIR} \\
-DCMAKE_C_COMPILER={ROOT_DIR}/bin/gcc \\
-DCMAKE_CXX_COMPILER={ROOT_DIR}/bin/g++ \\
-DCMAKE_CUDA_COMPILER={CUDA_PATH}/bin/nvcc \\
-DUSE_CUDSS={USE_CUDSS} \\
-DCAFFE2_USE_CUDNN={USE_CUDNN} \\
-DCAFFE2_USE_CUSPARSELT={USE_CUSPARSELT} \\
-DCAFFE2_USE_CUFILE=1 \\
-DCMAKE_PREFIX_PATH=\"{';'.join(_cmake_paths)};$CMAKE_PREFIX_PATH\" \\
-DCMAKE_BUILD_TYPE={"Debug" if DEBUG else "Release"} \\
-Wno-dev
if [ $? -ne 0 ]; then
rm -rf objdir lib 
exit 1
fi
echo ""

make -j$(nproc)
if [ $? -ne 0 ]; then
rm -rf objdir lib
exit 1
fi
echo ""

rm -rf objdir
'''
            with open(file=os.path.join(CURRENT_DIR, "build.sh"), mode="w") as f: f.write(_command)
            log.debug(f"Build script is generated at: {CURRENT_DIR}/build.sh\n\n")
            os.system(command=f"chmod +x {CURRENT_DIR}/build.sh")
            ret = os.system(command=f"bash {CURRENT_DIR}/build.sh")
            shutil.rmtree(path=os.path.join(CURRENT_DIR, "objdir"))
            os.remove(path=os.path.join(CURRENT_DIR, "build.sh"))
            if ret != 0:
                shutil.rmtree(path=_lib_installation_path)
                raise RuntimeError(f"Shared library build failed for the module")
            with open(file=os.path.join(_lib_installation_path, "build_info.json"), mode="w") as f:
                json.dump(obj={
                            "OS":         f"{OS}-{OS_ARCH}",
                            "Python":     str(PYTHON_VERSION),
                            "CUDA":       str(CUDA_VERSION),
                            "Libtorch":   str(TORCH_VERSION),
                            "CUDNN":      str(CUDNN_VERSION) if USE_CUDNN else None,
                            "CUDSS":      USE_CUDSS if USE_CUDSS else None,
                            "CUSPARSELT": USE_CUSPARSELT if USE_CUSPARSELT else None,
                            "NVTX":       USE_NVTX if USE_NVTX else None},
                        fp=f, indent=2)
            integrity_sign_dir(directory_path=os.path.join(CURRENT_DIR, "src"),
                               hash_file_path=os.path.join(CURRENT_DIR, 'debug' if DEBUG else 'release', '.src.sha256'))
            integrity_sign_dir(directory_path=os.path.join(CURRENT_DIR, "include"),
                               hash_file_path=os.path.join(CURRENT_DIR, 'debug' if DEBUG else 'release', '.include.sha256'))
    import importlib.util
    from .system.path_tree import PathTree
    _lib_path = os.path.dirname(p=os.path.abspath(path=__file__))
    _lib_path = os.path.join(_lib_path, "debug" if DEBUG else "release", "lib", "libm00nny_utils.so")
    log.debug(f"Loading {_lib_path}...")
    module_name = _lib_path.split(".")[0].replace("lib", "")
    module_name = "m00nny_utils"
    log.debug(f"Searching {module_name}...")
    spec = importlib.util.spec_from_file_location(name=f"{module_name}", location=_lib_path)
    module = importlib.util.module_from_spec(spec=spec)
    spec.loader.exec_module(module=module)
__bootstrap__()