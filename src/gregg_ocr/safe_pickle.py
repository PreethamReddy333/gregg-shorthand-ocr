"""Load third-party pickles without letting them run arbitrary code.

The Topaz line-crop files are pickles. A plain ``pickle.load`` can execute
anything the file asks for, so only the numpy constructors needed to rebuild
arrays are allowed; any other import in the file raises instead of running.
"""

import pickle
from pathlib import Path

_ALLOWED = {
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked import {module}.{name}")


def load(path: Path):
    with open(path, "rb") as f:
        return _NumpyOnlyUnpickler(f).load()
