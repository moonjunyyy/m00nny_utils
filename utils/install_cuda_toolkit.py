def install_cuda_toolkit(version: str, installation_path: str) -> None:
    import os
    import tarfile
    import hashlib
    import requests

    if not "." in version: version = f"{version[:2]}.{version[2:]}" # Make the version into the format of 11.0
    cuda_distribution_base_url = "https://developer.download.nvidia.com/compute/cuda/redist/"
    # There may not be needed version, so we need to check the version.
    response = requests.get(url=f"{cuda_distribution_base_url}") # There is version json file list in html.
    if response.status_code != 200: raise ConnectionError(f"Failed to connect to the server: {cuda_distribution_base_url}")
    available_versions = []
    for line in response.text.split("\n"):
        dist_version = line.split(sep="redistrib_")[-1].split(sep=".json")[0]
        if dist_version and version in dist_version: available_versions.append(dist_version)
    if not available_versions: raise ValueError(f"Version {version} is not available in the server: {cuda_distribution_base_url}")
    print(f"Available versions: {available_versions}")
    available_versions.sort() # Get the latest version.
    version = available_versions[-1]
    response = requests.get(url=f"{cuda_distribution_base_url}/redistrib_{version}.json")
    if response.status_code != 200: raise ConnectionError(f"Failed to connect to the server: {cuda_distribution_base_url}/redistrib_{version}.json")
    cuda_installation_json = response.json()

    print(f"")
    operating_system = os.uname().sysname
    arch = os.uname().machine
    target_system = f"{operating_system}-{arch}".lower()
    print(f"Architecture: {target_system}")

    cuda_release_date = cuda_installation_json.pop('release_date')
    cuda_version = cuda_installation_json.pop('release_label')
    cuda_lable = cuda_installation_json.pop('release_product')

    print(f"Installing {cuda_lable}-{cuda_version} ({cuda_release_date})")
    print(f"Installation path: {installation_path}")
    if os.path.exists(path=installation_path):
        print(f"Installation path already exists.")
    else:
        print(f"Installation path created.")
        os.makedirs(name=installation_path)

    for key, value in cuda_installation_json.items():
        print()
        print(f"Package:      {key}")
        print(f"Name:         {value['name']}")
        print(f"Licence:      {value['license']}")
        print(f"Licence Path: {value['license_path']}")
        print(f"Version:      {value['version']}")
        if value['license_path'] != "CUDA Toolkit": print("Pass"); continue
        try: target_system_dict = value[target_system]
        except KeyError: print(f"{key} is not available for {target_system}"); continue
        relative_path = target_system_dict['relative_path']
        sha256        = target_system_dict['sha256']
        md5           = target_system_dict['md5']
        size      = int(target_system_dict['size'])
        print(f"Relative Path: {relative_path}")
        print(f"SHA256:        {sha256}")
        print(f"MD5:           {md5}")
        if   size > 1024**3:print(f"Size:          {size/1024**3:.2f} GB")
        elif size > 1024**2:print(f"Size:          {size/1024**2:.2f} MB")
        elif size > 1024:   print(f"Size:          {size/1024:.2f} KB")
        else:               print(f"Size:          {size} B")
        print(f"Downloading    {key}...")
        url = f"{cuda_distribution_base_url}{relative_path}"
        print(f"URL: {url}")
        response = requests.get(url=url)
        if response.status_code != 200:
            print(f"Failed to download {key}")
            exit(code=1)
        file_name = os.path.basename(p=relative_path.split('/')[-1])
        file_path = os.path.join(installation_path, file_name)
        print(f"Saving to {file_path}")
        with open(file=file_path, mode='wb') as f:
            f.write(response.content)
        print(f"Downloaded {key}")
        print(f"Verifying {key}...", end="\r")
        with open(file=file_path, mode='rb') as f:
            file_content = f.read()
            sha256_hash = hashlib.sha256(string=file_content).hexdigest()
            md5_hash = hashlib.md5(string=file_content).hexdigest()
        if sha256_hash != sha256:
            print(f"SHA256 verification failed!")
            exit(code=1)
        if md5_hash != md5:
            print(f"MD5 verification failed!")
            exit(code=1)
        print(f"Verified {key}     ")
        print(f"Extracting {key}...", end="\r")
        archive_type = file_name.split('.')[-1]
        if archive_type == 'gz':
            with tarfile.open(name=file_path, mode='r:gz') as tar:
                tar.extractall(path=installation_path)
        elif archive_type == 'xz':
            with tarfile.open(name=file_path, mode='r:xz') as tar:
                tar.extractall(path=installation_path)
        elif archive_type == 'bz2':
            with tarfile.open(name=file_path, mode='r:bz2') as tar:
                tar.extractall(path=installation_path)
        elif archive_type == 'tar':
            with tarfile.open(name=file_path, mode='r') as tar:
                tar.extractall(path=installation_path)
        else:
            print(f"Unsupported archive type!")
            continue
        print(f"Extracted {key}     ")
         
        print(f"Moving {key}...", end="\r")
        extracted_folder_name = os.path.basename(p=file_name).split('.tar')[0]
        extracted_folder_path = os.path.join(installation_path, extracted_folder_name)
        for _dir in os.listdir(path=extracted_folder_path):
            if not os.path.exists(path=os.path.join(installation_path, _dir)): os.makedirs(name=os.path.join(installation_path, _dir))
            for _item in os.listdir(path=os.path.join(extracted_folder_path, _dir)):
                os.rename(src=os.path.join(extracted_folder_path, _dir, _item), dst=os.path.join(installation_path, _dir, _item))
        print(f"Moved {key}     ", end="\r")
        print(f"Cleaning up {key}...", end="\r")
        os.remove(path=file_path)
        os.rmdir(path=extracted_folder_path)
        print(f"Cleaned up {key}     ")