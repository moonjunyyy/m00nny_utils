from . import algorithm
from . import linalg
from .system.log import Log as Log


def __bootstrap__() -> None:
    import multiprocessing as mp

    import os
    from .system.log import Log

    _log_level = os.environ.get("LIBM00NNY_LOG_LEVEL", "INFO")
    _log_path = os.environ.get("LIBM00NNY_LOG_PATH", None)
    if _log_path:
        _log_path = os.path.abspath(path=_log_path)
    log = Log(
        name="m00nny_utils_init",
        global_level=_log_level,
        path=_log_path
    )

    import sys
    import json
    import sysconfig
    import torch
    import shutil
    from .util.version import Version
    from .util.signdir import validate_signed_dir, sign_dir
    from .install.install_requirements import install_cuda_libs, install_nccl

    __lock = mp.Lock()
    FORCE_BUILD: int        = int(os.environ.get("LIBM00NNY_FORCE_REBUILD",  "0"))
    ROOT_DIR: str           = os.path.expanduser(os.path.abspath(os.environ.get("LIBM00NNY_ROOT_DIR",os.path.join(os.path.expanduser("~/.local/")))))
    CURRENT_DIR: str        = os.path.dirname(p=__file__)
    
    _os_info = os.uname()    
    OS: str                 = _os_info.sysname.lower()
    OS_VERSION: str         = _os_info.release
    OS_ARCH: str            = _os_info.machine.lower()
    
    _torch_version          = torch.__version__.split('+')[0]  # 1.13.1
    _cuda_version           = torch.version.cuda
    _torch_cudnn_version    = torch.backends.cudnn.version() # 90100
    
    PYTHON_PATH:    str     = sys.executable
    PYTHON_VERSION: Version = Version(sys.version)
    TORCH_VERSION:  Version = Version(_torch_version)
    CUDA_VERSION:   Version = Version(_cuda_version)
    CUDNN_VERSION:  Version = Version(
        _torch_cudnn_version // 10000,
        _torch_cudnn_version % 10000 // 100,
        _torch_cudnn_version % 100
    )

    # The native sources (csrc/) are a torch extension that calls none of cuDNN, cuDSS,
    # cuSPARSELt or NCCL; torch itself brings its own copies at run time (PyPI nvidia-* packages).
    # So these are off unless asked for (=1), and only the CUDA toolkit (nvcc) is installed.
    USE_CUDNN: int = int(os.environ.get("LIBM00NNY_USE_CUDNN",      "0"))
    USE_CUDSS: int = int(os.environ.get("LIBM00NNY_USE_CUDSS",      "0"))
    USE_CUSPARSELT: int = int(os.environ.get("LIBM00NNY_USE_CUSPARSELT", "0"))
    USE_NCCL: int       = int(os.environ.get("LIBM00NNY_USE_NCCL",       "0"))
    DEBUG: int          = int(os.environ.get("LIBM00NNY_DEBUG",          "0"))

    # Check the GCC, G++, CMake
    ret_code = os.system("which gcc > /dev/null 2>&1")
    log.debug(f"which gcc returned code: {ret_code}")
    assert ret_code == 0, "GCC is not installed. Please install GCC and set the PATH environment variable."
    ret_code = os.system("which g++ > /dev/null 2>&1")
    log.debug(f"which g++ returned code: {ret_code}")
    assert ret_code == 0, "G++ is not installed. Please install G++ and set the PATH environment variable."
    ret_code = os.system("which cmake > /dev/null 2>&1")
    log.debug(f"which cmake returned code: {ret_code}")
    assert ret_code == 0, "CMake is not installed. Please install CMake and set the PATH environment variable."
    # assert os.path.isfile(path=os.path.join(ROOT_DIR, "bin", "gcc")), "GCC path not found."
    # assert os.path.isfile(path=os.path.join(ROOT_DIR, "bin", "g++")), "G++ path not found."
    # assert os.path.isfile(path=os.path.join(ROOT_DIR, "bin", "cmake")), "CMake path not found."

    with __lock:
        _cmake_paths = []
        # Check the CUDA Toolkit
        CUDA_PATH = install_cuda_libs(
            lib_name="cuda",
            os_name=OS,
            os_arch=OS_ARCH,
            root_dir=ROOT_DIR,
            version=CUDA_VERSION,
            cuda_version=CUDA_VERSION,
            condition=lambda kv: kv[1]['license'] == "CUDA Toolkit"
            )
        assert os.path.isfile(
          path=os.path.join(
                CUDA_PATH,
                "bin",
                "nvcc"
            )
        ), "Nvcc path not found. Check the CUDA Toolkit installation."
        assert os.path.isfile(
            path=os.path.join(
                CUDA_PATH,
                "lib64",
                "libcudart.so"
            )
        ), "CUDA shared library path not found. Check the CUDA Toolkit installation."
        _cmake_paths.append(CUDA_PATH)

        # Libtorch: the torch this Python runs. The native sources link torch_python, which only an
        # installed torch has, and the built library is loaded into this torch, so it is built against
        # this torch's own CMake config rather than a separately downloaded libtorch.
        LIBTORCH_PATH = torch.utils.cmake_prefix_path
        assert os.path.isfile(
            path=os.path.join(LIBTORCH_PATH, "Torch", "TorchConfig.cmake")
        ), f"TorchConfig.cmake not found under {LIBTORCH_PATH}: this torch ({torch.__version__}) ships no CMake config."
        _cmake_paths.append(LIBTORCH_PATH)

        if USE_CUDNN:
            # Check the CuDNN
            CUDNN_PATH = install_cuda_libs(
                lib_name="cudnn",
                os_name=OS,
                os_arch=OS_ARCH,
                root_dir=ROOT_DIR,
                version=CUDNN_VERSION,
                cuda_version=CUDA_VERSION,
                condition=lambda kv: True
            )
            assert os.path.isfile(
                path=os.path.join( CUDNN_PATH, "lib", "libcudnn.so" )
            ), "CuDNN shared library path not found."
            _cmake_paths.append(CUDNN_PATH)
        if USE_CUDSS:
            # Check the CuDSS
            CUDSS_PATH = install_cuda_libs(
                lib_name="cudss",
                os_name=OS,
                os_arch=OS_ARCH,
                root_dir=ROOT_DIR,
                version=None,
                cuda_version=CUDA_VERSION,
                condition=lambda kv: True
            )
            assert os.path.isfile(
                path=os.path.join(CUDSS_PATH, "lib", "libcudss.so")
            ), "CuDSS shared library path not found."
            _cmake_paths.append(CUDSS_PATH)
        if USE_CUSPARSELT:
            # Check the CuSparseLT
            CUSPARSELT_PATH = install_cuda_libs(
                lib_name="cusparselt",
                os_name=OS,
                os_arch=OS_ARCH,
                root_dir=ROOT_DIR,
                version=None,
                cuda_version=CUDA_VERSION,
                condition=lambda kv: True
            )
            assert os.path.isfile(
                path=os.path.join(CUSPARSELT_PATH, "lib", "libcusparseLt.so" )
            ), "CuSparseLT shared library path not found."
            _cmake_paths.append(CUSPARSELT_PATH)
        if USE_NCCL:
            # Check the NCCL
            NCCL_PATH = install_nccl(
                os_name=OS,
                os_arch=OS_ARCH,
                root_dir=ROOT_DIR,
                cuda_version=CUDA_VERSION,
            )
            assert os.path.isfile(
                path=os.path.join(
                    NCCL_PATH,
                    "lib",
                    "libnccl.so"
                )
            ), f"NCCL shared library not found: {os.path.join(NCCL_PATH, 'lib', 'libnccl.so')}"
            _cmake_paths.append(NCCL_PATH)
        # What the built library depends on: written after a build, compared before reusing it.
        _build_info = {
            "OS":         f"{OS}-{OS_ARCH}",
            "Python":     str(PYTHON_VERSION),
            # the ABI the library is compiled for (e.g. cpython-314t-…: free-threaded), not told by the version
            "PythonABI":  sysconfig.get_config_var("SOABI"),
            "CUDA":       str(CUDA_VERSION),
            "Libtorch":   str(TORCH_VERSION),
            "CUDNN":      str(CUDNN_VERSION) if USE_CUDNN else None,
            "CUDSS":      USE_CUDSS if USE_CUDSS else None,
            "CUSPARSELT": USE_CUSPARSELT if USE_CUSPARSELT else None,
            "NCCL":       USE_NCCL if USE_NCCL else None,
        }
        # The native source trees the build compiles and signs, and where the built library lives.
        NATIVE_SOURCE_DIRS = ("csrc",)
        _lib_installation_path = os.path.join(CURRENT_DIR, "debug" if DEBUG else "release")
        try:
            if FORCE_BUILD: log.info("Forcing to build the shared library..."); raise RuntimeError("Forcing to build the shared library...")
            if not os.path.isdir(s=_lib_installation_path): os.makedirs(name=_lib_installation_path); raise RuntimeError("Shared library is not found.")
            if os.path.isdir(s=_lib_installation_path) and os.path.isdir(s=os.path.join(_lib_installation_path, "lib")):
                with open(file=os.path.join(_lib_installation_path, "build_info.json"), mode="r") as f:
                    _info = json.load(fp=f)
                if any(_info.get(k) != v for k, v in _build_info.items()):
                    raise RuntimeError("Shared library is not up-to-date.")

            if not all(validate_signed_dir(directory_path=os.path.join(CURRENT_DIR, _dir),
                                           hash_file_path=os.path.join(_lib_installation_path, f".{_dir}.sha256"))
                       for _dir in NATIVE_SOURCE_DIRS):
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
            log.debug(f"{'NCCL info:':<20}{USE_NCCL if USE_NCCL else None}")
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
-DPython_EXECUTABLE={PYTHON_PATH} \\
-DCMAKE_CUDA_COMPILER={CUDA_PATH}/bin/nvcc \\
-DUSE_CUDSS={USE_CUDSS} \\
-DCAFFE2_USE_CUDNN={USE_CUDNN} \\
-DCAFFE2_USE_CUSPARSELT={USE_CUSPARSELT} \\
-DCAFFE2_USE_CUFILE=0 \\
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
            with open(
                file=os.path.join(CURRENT_DIR, "build.sh"),
                mode="w"
            ) as f:
                f.write(_command)
            log.debug(f"Build script is generated at: {CURRENT_DIR}/build.sh\n\n")
            os.system(command=f"chmod +x {CURRENT_DIR}/build.sh")
            ret = os.system(command=f"bash {CURRENT_DIR}/build.sh")
            shutil.rmtree(path=os.path.join(CURRENT_DIR, "objdir"))
            os.remove(path=os.path.join(CURRENT_DIR, "build.sh"))
            if ret != 0:
                shutil.rmtree(path=_lib_installation_path)
                raise RuntimeError(f"Shared library build failed for the module")
            with open(
                file=os.path.join(_lib_installation_path, "build_info.json"),
                mode="w"
            ) as f:
                json.dump(obj=_build_info, fp=f, indent=2)
            for _dir in NATIVE_SOURCE_DIRS:
                sign_dir(directory_path=os.path.join(CURRENT_DIR, _dir),
                         hash_file_path=os.path.join(_lib_installation_path, f".{_dir}.sha256"))
    import importlib.util
    _lib_path = os.path.dirname(p=os.path.abspath(path=__file__))
    _lib_path = os.path.join(
        _lib_path,
        "debug" if DEBUG else "release",
        "lib",
        "libm00nny_utils.so"
    )
    log.debug(f"Loading {_lib_path}...")
    module_name = "m00nny_utils"
    spec = importlib.util.spec_from_file_location(
        name=f"{module_name}",
        location=_lib_path
    )
    module = importlib.util.module_from_spec(spec=spec)
    spec.loader.exec_module(module=module)


# U1/U0 split (2026-10-30): the stdlib-only core (system/, util/ formatting, version, ...)
# needs no native build. LIBM00NNY_SKIP_NATIVE=1 skips the import-time CUDA/libtorch build;
# unset, behaviour is unchanged.
import os as _os
if _os.environ.get("LIBM00NNY_SKIP_NATIVE", "0") != "1":
    try:
        import torch as _torch
        if getattr(_torch.version, "cuda", None) is None:
            raise ImportError(
                f"m00nny_utils: the import-time native build needs a CUDA build of torch, but this torch "
                f"({_torch.__version__}) has none. On Linux with an NVIDIA GPU, install the 'native' extra "
                "without a backend extra (uv sync --extra native: PyPI's torch is a CUDA build), "
                "or set LIBM00NNY_SKIP_NATIVE=1 to use the pure-Python core."
            )
        del _torch
        __bootstrap__()
    except ModuleNotFoundError as _e:
        raise ImportError(
            f"m00nny_utils: the import-time native build needs '{_e.name}', which is not installed. "
            "Install the 'native' extra (uv sync --extra native, on Linux with an NVIDIA GPU), "
            "or set LIBM00NNY_SKIP_NATIVE=1 to use the pure-Python core."
        ) from _e
del _os
