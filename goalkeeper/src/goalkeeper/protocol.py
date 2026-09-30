"""NumPy/msgpack wire format used by the official v3.9.2 evaluator.

No pickle and no object arrays. reset=True receives NO reply, as in the official
client; replying to reset would desynchronize its next inference response.
"""
import numpy as np
import msgpack


def _pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf": raise ValueError("Unsupported array dtype")
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biuf": raise ValueError("Unsupported scalar dtype")
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"Cannot serialize {type(value)}")


def _unpack(value):
    if b"__ndarray__" in value:
        dtype, shape = np.dtype(value[b"dtype"]), tuple(value[b"shape"])
        if dtype.kind not in "biuf" or any(type(s) is not int or s < 0 for s in shape):
            raise ValueError("Invalid ndarray encoding")
        import math
        if math.prod(shape) * dtype.itemsize != len(value[b"data"]):
            raise ValueError("Array byte count mismatch")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(shape).copy()
    if b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "biuf": raise ValueError("Invalid scalar dtype")
        return dtype.type(value[b"data"])
    return value


def packb(value): return msgpack.packb(value, default=_pack, use_bin_type=True)
def unpackb(value): return msgpack.unpackb(value, object_hook=_unpack, strict_map_key=False)
