def count_digits(number) -> int:
    return len(str(object=number))


def bytes_to_human_readable(size: int, precision=2) -> str:
    """
    Convert bytes to human readable format.
    {size} -> {size} {suffix}

    Args:
        size (int):
            The number of bytes to convert.
        precision (int, optional):
            The number of decimal places to round.
            Defaults to 2.

    Returns (str):
        The human readable format of the bytes.
    """
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    suffix_index = 0
    while size > 1024 and suffix_index < 5:
        suffix_index += 1
        size /= 1024
    return f"{size:.{precision}f} {suffixes[suffix_index]}"


def seconds_to_human_readable(seconds: int) -> str:
    """
    Convert seconds to human readable format.
    {seconds} -> {h}:{m}:{s}.{ms} or {m}:{s}.{ms} or {s}.{ms} or {ms}

    Args:
        seconds (int):
            The number of seconds to convert.

    Returns (str):
        The human readable format of the seconds.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    f = seconds - int(seconds)
    hms = ""
    if h != 0:
        hms += f"{h:02}:"
    if hms != "" or m != 0:
        hms += f"{m:02}:"
    if hms != "" or s != 0:
        hms += f"{s:02}"
    if hms == "":
        hms = f"{int(f*1000):3d}ms"
    else:
        hms += f".{int(f*1000):03d}"
    return hms
