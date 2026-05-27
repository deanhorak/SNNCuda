from __future__ import annotations

import ctypes
import ctypes.util
import os
import pathlib
from typing import Optional


class SNNCudaLibraryError(RuntimeError):
    pass


def _candidate_paths() -> list[str]:
    candidates: list[str] = []
    explicit = os.environ.get("SNNCUDA_LIBRARY")
    if explicit:
        candidates.append(explicit)

    package_dir = pathlib.Path(__file__).resolve().parent
    for name in ("libsnncuda.so", "libsnncuda.dylib", "snncuda.dll"):
        candidates.append(str(package_dir / name))

    discovered = ctypes.util.find_library("snncuda")
    if discovered:
        candidates.append(discovered)
    return candidates


def _load_library() -> ctypes.CDLL:
    errors: list[str] = []
    for candidate in _candidate_paths():
        try:
            library = ctypes.CDLL(candidate)
            library.snncuda_version.restype = ctypes.c_char_p
            library.snncuda_cuda_available.restype = ctypes.c_int
            return library
        except OSError as error:
            errors.append(f"{candidate}: {error}")

    details = "\n".join(errors) if errors else "no candidate library paths found"
    raise SNNCudaLibraryError(
        "Unable to load SNNCuda shared library. Set SNNCUDA_LIBRARY to the "
        f"installed libsnncuda path.\n{details}"
    )


_library: Optional[ctypes.CDLL] = None


def _lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = _load_library()
    return _library


def version() -> str:
    return _lib().snncuda_version().decode("utf-8")


def cuda_available() -> bool:
    return bool(_lib().snncuda_cuda_available())


__all__ = ["SNNCudaLibraryError", "cuda_available", "version"]
