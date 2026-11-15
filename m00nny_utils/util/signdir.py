from ..system.log import Log

log = Log(name="integrity_verify")


def get_file_hash(file_path: str, hash_type: str = "sha256") -> str:
    import hashlib

    if hash_type == "sha256":
        hash_obj = hashlib.sha256()
    elif hash_type == "sha1":
        hash_obj = hashlib.sha1()
    elif hash_type == "md5":
        hash_obj = hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash type: {hash_type}")
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def validate_signed_file(
    file_path: str, hash_type: str = "sha256", hash_file_path: str = None
) -> None:
    import os

    if hash_file_path is None:
        hash_file_path = os.path.join(file_path, ("." + hash_type))
    if not os.path.isfile(hash_file_path):
        raise FileNotFoundError(f"Hash file not found: {hash_file_path}")
    with open(hash_file_path, "r") as f:
        hash_value = f.read()
    if get_file_hash(file_path=file_path, hash_type=hash_type) == hash_value:
        return True


def _signed_files(directory_path: str, hash_file_path: str) -> dict:
    """The files a directory signature covers, as {rel_path: path}: every file under the directory except
    hidden entries (a path component starting with '.': OS or sync litter such as .DS_Store, ._*, .nfs*)
    and the signature file itself. sign_dir writes exactly this set and validate_signed_dir compares against
    exactly this set, so a file added after signing counts as a change, like an edited or a removed one."""
    import os
    from ..system.path_tree import PathTree

    _signature = os.path.realpath(hash_file_path)
    return {
        _path.rel_path: _path.path
        for _path in PathTree(directory_path, depth_limit=10)
        if _path.is_file
        and not any(_part.startswith(".") for _part in _path.rel_path.split(os.sep))
        and os.path.realpath(_path.path) != _signature
    }


def validate_signed_dir(
    directory_path: str, hash_type: str = "sha256", hash_file_path: str = None
) -> None:
    import os

    if hash_file_path is None:
        hash_file_path = os.path.join(directory_path, ("." + hash_type))
    try:
        if not os.path.isfile(hash_file_path):
            raise FileNotFoundError(f"Hash file not found: {hash_file_path}")
        _signed = {}
        with open(hash_file_path, "r") as f:
            for line in f:
                if line.strip():
                    _file, _hash = line.rsplit(sep=":", maxsplit=1)
                    _signed[_file.strip()] = _hash.strip()
        _current = _signed_files(directory_path=directory_path, hash_file_path=hash_file_path)
        _changes = {
            "added":   sorted(_current.keys() - _signed.keys()),
            "missing": sorted(_signed.keys() - _current.keys()),
            "changed": sorted(
                _rel for _rel in _current.keys() & _signed.keys()
                if get_file_hash(file_path=_current[_rel], hash_type=hash_type) != _signed[_rel]
            ),
        }
        if any(_changes.values()):
            raise ValueError("; ".join(f"{_kind}: {_rels}" for _kind, _rels in _changes.items() if _rels))
        return True
    except Exception as e:
        log.warning(
            f"Error occurred while verifying the integrity of the directory:\n{e}"
        )
    return False


def sign_file(
    file_path: str, hash_type: str = "sha256", hash_file_path: str = None
) -> None:
    import os

    if hash_file_path is None:
        hash_file_path = os.path.join(file_path, ("." + hash_type))
    with open(hash_file_path, "w") as f:
        f.write(f"{get_file_hash(file_path=file_path, hash_type=hash_type)}")
    log.debug(f"Integrity signature saved: {hash_file_path}")


def sign_dir(
    directory_path: str, hash_type: str = "sha256", hash_file_path: str = None
) -> None:
    import os

    if hash_file_path is None:
        hash_file_path = os.path.join(directory_path, ("." + hash_type))
    _files = _signed_files(directory_path=directory_path, hash_file_path=hash_file_path)
    with open(hash_file_path, "w") as f:
        for _rel in sorted(_files):
            log.debug(f"Signing: {_rel}")
            f.write(f"{_rel}: {get_file_hash(file_path=_files[_rel], hash_type=hash_type)}\n")
    log.debug(f"Integrity signature saved: {hash_file_path}")
