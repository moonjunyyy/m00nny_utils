def get_unique_project_id(salt: str = "", pepper: str = "") -> str:
    '''
    Generate a unique project identifier from the current project directory.
    The identifier is hashed and salted and peppered to prevent collision.
    (The version or conditions of the project can be added as salt or pepper.)
    
    Args:
        salt   (str, optional): Salt to add to the hash value. Defaults to "".
        pepper (str, optional): Pepper to add to the hash value. Defaults to "".

    Returns:
        str: Unique project identifier. (128 Hexadecimal characters)
    '''
    import os
    import inspect 
    import hashlib
    # Generate from current project directory unique hashed identifier.
    _dir_path = os.path.dirname(inspect.stack()[-1].filename)
    print(f"Project Directory: {_dir_path}")
    print(f"Salt:              {salt}")
    print(f"Pepper:            {pepper}")
    _hash_val           = hashlib.sha3_512(os.path.dirname(inspect.stack()[-1].filename).encode()).hexdigest()
    _salted_hash_val    = hashlib.sha3_512((_hash_val + salt).encode()).hexdigest()
    _peppered_hash_val  = hashlib.sha3_512((_salted_hash_val + pepper).encode()).hexdigest()
    # the length of the hash value is 128 (512 bits)
    # Get the Integer value of the each 2 characters of the hash value. (-128 ~ 127)
    _hash = [int(_hash_val[i:i+2], 16) - 128 for i in range(0, 128, 2)]
    _hash = "".join([_hash_val[i] for i in _hash])
    print(f"Hash Value:        {_hash}")
    return _hash