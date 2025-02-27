from ..system.log import Log
log = Log(name="integrity_verify")

def get_file_hash(file_path: str, hash_type: str = "sha256") -> str:
    import hashlib
    if hash_type == "sha256": hash_obj = hashlib.sha256()
    elif hash_type == "sha1": hash_obj = hashlib.sha1()
    elif hash_type == "md5":  hash_obj = hashlib.md5()
    else: raise ValueError(f"Unsupported hash type: {hash_type}")
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def integrity_verify(file_path: str, hash_type: str = "sha256", hash_file_path: str = None) -> None:
    import os
    if hash_file_path is None: hash_file_path = os.path.join(file_path, ('.' + hash_type))
    if not os.path.isfile(hash_file_path):
        raise FileNotFoundError(f"Hash file not found: {hash_file_path}")
    with open(hash_file_path, "r") as f: hash_value = f.read()
    if get_file_hash(file_path=file_path, hash_type=hash_type) == hash_value:
        return True
    
def integrity_verify_dir(directory_path: str, hash_type: str = "sha256", hash_file_path: str = None) -> None:
    import os
    if hash_file_path is None: hash_file_path = os.path.join(directory_path, ('.' + hash_type))
    hash_dict = {}
    try:
        if not os.path.isfile(hash_file_path):
            raise FileNotFoundError(f"Hash file not found: {hash_file_path}")
        with open(hash_file_path, "r") as f:
            for line in f.readlines():
                _file, _hash = line.split(sep=":")
                hash_dict[_file.strip()] = _hash.strip()
        from ..system.path_tree import PathTree
        path_tree = PathTree(directory_path, depth_limit=10)
        for _path in path_tree:
            if _path.name == ('.' + hash_type): continue
            if _path.is_file() and _path.rel_path in hash_dict:
                log.debug(f"Verifying: {_path.rel_path}")
                if get_file_hash(file_path=_path.path, hash_type=hash_type) != hash_dict.pop(_path.rel_path, None):
                    raise ValueError(f"{_path.rel_path} expected hash: {hash_dict[_path.rel_path]} but got different hash")
        if len(hash_dict) > 0: raise ValueError(f"Following files are missing: {hash_dict.keys()}")
        return True
    except Exception as e: log.warning(f"Error occurred while verifying the integrity of the directory:\n{e}")
    return False

def integrity_sign(file_path: str, hash_type: str = "sha256", hash_file_path: str = None) -> None:
    import os
    if hash_file_path is None: hash_file_path = os.path.join(file_path, ('.' + hash_type))
    with open(hash_file_path, "w") as f:
        f.write(f"{get_file_hash(file_path=file_path, hash_type=hash_type)}")
    log.debug(f"Integrity signature saved: {hash_file_path}")

def integrity_sign_dir(directory_path: str, hash_type: str = "sha256", hash_file_path: str = None) -> None:
    import os
    if hash_file_path is None: hash_file_path = os.path.join(directory_path, ('.' + hash_type))
    with open(hash_file_path, "w") as f:
        from ..system.path_tree import PathTree
        path_tree = PathTree(directory_path, depth_limit=10)
        for _path in path_tree:
            log.debug(f"Signing: {_path.rel_path}")
            if _path.name == ('.' + hash_type): continue
            if _path.is_file():
                f.write(f"{_path.rel_path}: {get_file_hash(file_path=_path.path, hash_type=hash_type)}\n")
    log.debug(f"Integrity signature saved: {hash_file_path}")