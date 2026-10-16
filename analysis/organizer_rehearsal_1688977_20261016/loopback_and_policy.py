"""Bring up only the private namespace loopback, drop capabilities, start policy."""
import ctypes
import fcntl
import os
import socket
import struct

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    flags = fcntl.ioctl(sock.fileno(), 0x8913, struct.pack("16sh", b"lo", 0))
    value = struct.unpack("16sh", flags[:18])[1]
    fcntl.ioctl(sock.fileno(), 0x8914, struct.pack("16sh", b"lo", value | 1))


class Header(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class Caps(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]


libc = ctypes.CDLL(None, use_errno=True)
header = Header(0x20080522, 0)
caps = (Caps * 2)()
if libc.capset(ctypes.byref(header), ctypes.byref(caps)) != 0:
    raise OSError(ctypes.get_errno(), "Cannot drop bootstrap capabilities")
os.execv("/opt/behavior/serve.sh", ["/opt/behavior/serve.sh"])
