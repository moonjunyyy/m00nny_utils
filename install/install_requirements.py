import os
import shutil
import requests
from typing import Callable
from ..system.path_tree import PathTree
from ..util.version import Version
from ..util.download_url import DownloadURL
from ..util.signdir import get_file_hash

from ..system.log import Log
log = Log(name="install_requirements")

def install_cuda_libs(lib_name      : str,
                      os_name       : str,
                      os_arch       : str,
                      root_dir      : str,
                      version       : Version,
                      cuda_version  : Version,
                      condition     : Callable = lambda kv : True)-> str:

        os_name = os_name.lower()
        os_arch = os_arch.lower()
        # Check for the prior installation (root_dir/{lib_name}*)
        _prior_installation = [Version(f.split(f"{lib_name}-")[-1]) for f in os.listdir(root_dir) if f.startswith(f"{lib_name}-")]
        for _prior in _prior_installation:
            if version is None:
                log.info(f"Found the {lib_name} installation: {os.path.join(root_dir, f"{lib_name}-{_prior}")}")
                return os.path.join(root_dir, f"{lib_name}-{_prior}")
            if version is not None and _prior == version:
                log.info(f"Found the {lib_name} installation: {os.path.join(root_dir, f"{lib_name}-{_prior}")}")
                return os.path.join(root_dir, f"{lib_name}-{_prior}")

        _base_url = f"https://developer.download.nvidia.com/compute/{lib_name}/redist"
        _res = requests.get(url=_base_url)
        if _res.status_code != 200: raise ConnectionError(f"Failed to connect to the server: {_base_url}")
        _versions = []
        for _line in _res.text.split(sep="\n"):
            if not "redistrib_" in _line: continue
            _json_ver = _line.split(sep="href=\"")[-1].split(sep="\">")[0]
            _json_ver = Version(_json_ver.split(sep="redistrib_")[-1].split(sep=".json")[0])
            if version is not None and _json_ver != version: continue
            _versions.append(_json_ver)
        _versions.sort()
        _maximum_available_version = _versions[-1]

        # Download the redist{version}.json file
        _installation_path = os.path.join(root_dir, f"{lib_name}-{_maximum_available_version}")
        if not os.path.isdir(s=_installation_path): os.makedirs(name=_installation_path)
        _installation_path_tree = PathTree(path=_installation_path, depth_limit=20)
        try:
            _json_url = f"https://developer.download.nvidia.com/compute/{lib_name}/redist/redistrib_{_maximum_available_version}.json"
            _res = requests.get(url=_json_url)
            if _res.status_code != 200: raise ConnectionError(f"Failed to connect to the server: {_json_url}")
            _json = _res.json()
            
            _release_date = _json.pop('release_date',    None)
            _version      = _json.pop('release_label',   None)
            _lable        = _json.pop('release_product', None)
            log.info(f"Installing {_lable}-{_version} ({_release_date})")
            for key, value in _json.items():
                log.debug(f"{'Package:':<20}{key}")
                log.debug(f"{'Name:':<20}{value['name']}")
                log.debug(f"{'Licence:':<20}{value['license']}")
                log.debug(f"{'Licence Path:':<20}{value['license_path']}")
                log.debug(f"{'Version:':<20}{value['version']}")
                if not condition((key, value)): log.info(f"Skipping the {key} installation."); continue
                try:
                    _cuda_variants = value.pop('cuda_variant', None)
                    log.debug(f"{'OS Variant:':<20}{os_name}-{os_arch}")
                    if _cuda_variants is not None: 
                        log.debug(f"{'CUDA Variant:':<20}cuda{cuda_version.major}")
                        _lib_for_arch = value[f'{os_name}-{os_arch}'][f'cuda{cuda_version.major}']
                    else: _lib_for_arch = value[f'{os_name}-{os_arch}']
                except KeyError as e:
                    log.info(f"Failed to find the {os_name}-{os_arch} variant for cuda{cuda_version.major}, Skipping the installation.")
                    continue
                _relative_path = _lib_for_arch['relative_path']
                _sha256        = _lib_for_arch['sha256']
                _md5           = _lib_for_arch['md5']
                _size          = _lib_for_arch['size']
                _filename = _relative_path.split(sep="/")[-1]
                _download_url = f"https://developer.download.nvidia.com/compute/{lib_name}/redist/{_relative_path}"
                _download_path = os.path.join(_installation_path, f"{_filename}")
                DownloadURL(url=_download_url, save_path=os.path.join(_download_path)).download()
                if not get_file_hash(file_path=_download_path, hash_type="sha256") == _sha256:
                    raise ValueError(f"Failed to verify the SHA256 Signature: {_download_path}")
                if not get_file_hash(file_path=_download_path, hash_type="md5") == _md5:
                    raise ValueError(f"Failed to verify the MD5 Signature: {_download_path}")
                os.system(command=f"tar -xf {_download_path} -C {_installation_path}")
                os.remove(path=os.path.join(_download_path))
                _extracted_path = _filename.split(sep=".tar")[0]
                _extracted_path_tree = PathTree(path=os.path.join(_installation_path, _extracted_path), depth_limit=20)
                log.debug(f"{'Extracted Path:':<20}\n{_extracted_path_tree}")
                while True:
                    if len(_extracted_path_tree) == 0:
                        _extracted_path_tree.rm()
                        break
                    _dir = _extracted_path_tree.children[0]
                    _dir.mv(dst=_installation_path_tree)
        except Exception as e:
            shutil.rmtree(path=_installation_path)
            raise RuntimeError(f"Failed to install the {key} in path: {_installation_path}\n{e}")
        if os.path.isdir(s=os.path.join(_installation_path, "lib64")):
            _lib64_dir = _installation_path_tree.get("lib64")
            _lib_dir   = _installation_path_tree.get("lib")
            for _item in _lib64_dir.children: _item.mv(dst=_lib_dir)
            _lib64_dir.rm()
        if os.path.isdir(s=os.path.join(_installation_path, "lib64")):
            _installation_path_tree.get("lib64").merge(dst=_installation_path_tree.get("lib"))
        os.symlink(src=os.path.join(_installation_path, "lib"), dst=os.path.join(_installation_path, "lib64"))
        return _installation_path

def install_nvtx(root_dir : str) -> str:
    import re
    _nvtx_list_url = "https://github.com/NVIDIA/NVTX/tags"
    pattern = re.compile(r'<a href="/NVIDIA/NVTX/releases/tag/v(\d+\.\d+\.\d+)"[^>]*>v(\d+\.\d+\.\d+)</a>')
    try:
        import requests
        _nvtx_list    = requests.get(url=_nvtx_list_url).text
        _nvtx_version = pattern.findall(string=_nvtx_list)
        if len(_nvtx_version) == 0: raise ValueError(f"No NVTX version found.")
        _nvtx_version.sort()
        _nvtx_version = _nvtx_version[-1]
        _nvtx_version = f"v{_nvtx_version}"
        _nvtx_url     = f"https://github.com/NVIDIA/NVTX/archive/refs/tags/{_nvtx_version}.tar.gz"
        _nvtx_base_path = os.path.join(root_dir, "third_party")
        if not os.path.isdir(s=os.path.join(_nvtx_base_path, "NVTX")):
            if not os.path.isdir(s=_nvtx_base_path): os.makedirs(name=_nvtx_base_path)
            # Download the NVTX
            if not os.path.isfile(path=os.path.join(root_dir, "v3.1.0.tar.gz")):
                DownloadURL(url=_nvtx_url, save_path=os.path.join(_nvtx_base_path, "v3.1.0.tar.gz")).download()
                os.system(command=f"tar -xf {_nvtx_base_path}/v3.1.0.tar.gz -C {_nvtx_base_path}")
                os.remove(path=os.path.join(_nvtx_base_path, "v3.1.0.tar.gz"))
                os.rename(src=os.path.join(_nvtx_base_path, "NVTX-3.1.0"), dst=os.path.join(_nvtx_base_path, "NVTX"))
        else: log.info(f"Found the NVTX.")
        return os.path.join(_nvtx_base_path, "NVTX")
    except Exception as e:
        shutil.rmtree(path=_nvtx_base_path)
        raise RuntimeError(f"Failed to install the NVTX.")


def install_libtorch(os_name   : str,
                     root_dir  : str,
                     version   : Version,
                     cuda_version: Version,
                     use_cxx11_abi:bool=False) -> str:
    for f in os.listdir(root_dir):
        if not f.startswith("libtorch_"): continue
        torch_ver, cuda_ver = f.split("libtorch_")[-1].split("_cu")
        if torch_ver == version and cuda_ver == f"{cuda_version.major}{cuda_version.minor}":
            log.info(f"Found the Libtorch installation: {os.path.join(root_dir, f)}")
            return os.path.join(root_dir, f)
    _libtorch_base_path = f"{root_dir}/libtorch_{version}_cu{cuda_version.major}{cuda_version.minor}"
    if not os.path.isdir(s=_libtorch_base_path): os.makedirs(name=_libtorch_base_path)
    try:
        _libtorch_list_url = f"https://download.pytorch.org/libtorch/cu{cuda_version.major}{cuda_version.minor}/"
        res = requests.get(url=_libtorch_list_url)
        if res.status_code != 200: raise ConnectionError(f"Failed to connect to the server: {_libtorch_list_url}")
        available_versions = []
        for line in res.text.split(sep="\n"):
            if "libtorch-" in line: available_versions.append(line.split(sep="href=\"")[-1].split(sep="\"")[0])
        if os_name == "Windows": available_versions = [_v for _v in available_versions if "win" in _v]
        elif use_cxx11_abi:
            abi_available_versions = [_v for _v in available_versions if "cxx11-abi" in _v     and "win" not in _v]
            abi_available_versions.sort()
            _libtorch_file_name = abi_available_versions[-1] # Get the latest version.
        else:
            abi_not_available_versions = [_v for _v in available_versions if "cxx11-abi" not in _v and "win" not in _v]
            abi_not_available_versions.sort()
            _libtorch_file_name = abi_not_available_versions[-1]
        log.info(f"Installing libtorch version: {_libtorch_file_name}")
        _libtorch_url = f"https://download.pytorch.org/{_libtorch_file_name.replace('+', '%2B')}"
        _libtorch_file_name = _libtorch_file_name.split(sep="/")[-1]
        DownloadURL(url=_libtorch_url, save_path=os.path.join(_libtorch_base_path, _libtorch_file_name)).download()
        os.system(command=f"unzip -q {os.path.join(_libtorch_base_path, _libtorch_file_name)} -d {_libtorch_base_path}")
        os.remove(path=os.path.join(_libtorch_base_path, _libtorch_file_name))
        if not os.path.isdir(s=os.path.join(_libtorch_base_path, "libtorch")):
            raise FileNotFoundError(f"Libtorch path not found: {_libtorch_base_path}/libtorch.\n"
                                    f"You may need to download the libtorch manually proper version from the link: {_libtorch_url}")
        for _item in os.listdir(path=os.path.join(_libtorch_base_path, "libtorch")):
            os.rename(src=os.path.join(_libtorch_base_path, "libtorch", _item), dst=os.path.join(_libtorch_base_path, _item))
        shutil.rmtree(path=os.path.join(_libtorch_base_path, "libtorch"))
    except Exception as e:
        shutil.rmtree(path=_libtorch_base_path)
        raise RuntimeError(f"Failed to install the Libtorch in path: {_libtorch_base_path}\n{e}")
    return _libtorch_base_path
