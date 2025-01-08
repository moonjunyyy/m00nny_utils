def install_libtorch(cuda_version: str, libtorch_version: str, installation_path: str) -> None:
    import os
    import zipfile
    import requests

    _libtorch_list_url = f"https://download.pytorch.org/libtorch/cu{cuda_version}/"
    res = requests.get(url=_libtorch_list_url)
    if res.status_code != 200: raise ConnectionError(f"Failed to connect to the server: {_libtorch_list_url}")
    available_versions = []
    for line in res.text.split(sep="\n"):
        if str(object=libtorch_version) in line:
            available_versions.append(line)
    abi_available_versions     = [_v for _v in available_versions if "cxx11-abi" in _v     and "win" not in _v]
    abi_not_available_versions = [_v for _v in available_versions if "cxx11-abi" not in _v and "win" not in _v]
    abi_available_versions.sort()
    abi_not_available_versions.sort()
    if abi_available_versions:       _libtorch_file_name = abi_available_versions[-1] # Get the latest version.
    elif abi_not_available_versions: _libtorch_file_name = abi_not_available_versions[-1] # Get the latest version.
    else: raise ValueError(f"Version {libtorch_version}+cu{cuda_version} is not available in the server: {_libtorch_list_url}")

    print(f"Libtorch version: {libtorch_version}")
    _libtorch_url = f"https://download.pytorch.org/libtorch/cu{cuda_version}/{_libtorch_file_name.replace('+', '%2B')}"
    res = requests.get(url=_libtorch_url)
    if res.status_code != 200: raise RuntimeError(f"Libtorch download failed with status code: {res.status_code}.\n"
                                                  f"Please check the libtorch url: {_libtorch_url}\n"
                                                  f"You may need to download the libtorch manually proper version from the link: {_libtorch_url}")
    with open(file=os.path.join(installation_path, _libtorch_file_name), mode="wb") as f:
        # Save the libtorch zip file
        f.write(res.content)
        # Extract the libtorch zip file to the base environment path (_base_env_path/libtorch)
        zipfile.ZipFile(file=f).extractall(path=installation_path)
    # Remove the libtorch zip file
    os.remove(path=os.path.join(installation_path, _libtorch_file_name))
    if not os.path.exists(path=os.path.join(installation_path, "libtorch")):
        raise FileNotFoundError(f"Libtorch path not found: {installation_path}/libtorch.\n"
                                f"You may need to download the libtorch manually proper version from the link: {_libtorch_url}")