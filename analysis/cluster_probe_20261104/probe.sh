echo PROBE_BEGIN; date; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nvidia-smi | head -4 | tail -1
echo "NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES"
ls -la /usr/lib/x86_64-linux-gnu/libEGL_nvidia* /usr/lib64/libEGL_nvidia* /usr/lib/x86_64-linux-gnu/libnvidia-eglcore* 2>&1 | head -6
ls /usr/share/glvnd/egl_vendor.d/ /etc/glvnd/egl_vendor.d/ 2>&1 | head -6
ls -la /usr/lib/x86_64-linux-gnu/libEGL.so* /usr/lib/x86_64-linux-gnu/libOSMesa* 2>&1 | head -4
nproc; free -g | head -2; df -h /mnt/afs /tmp / 2>&1 | tail -4
which python3; python3 --version; python3 -c "import ctypes; ctypes.CDLL('libEGL.so.1'); print('libEGL.so.1 loadable')" 2>&1 | tail -1
python3 -c "import ctypes,ctypes.util; l=ctypes.CDLL('libEGL.so.1'); f=l.eglGetProcAddress; f.restype=ctypes.c_void_p; f.argtypes=[ctypes.c_char_p]; print('eglQueryDevicesEXT', bool(f(b'eglQueryDevicesEXT')))" 2>&1 | tail -1
pip list 2>/dev/null | grep -i -E "^(torch|mujoco|pybullet|robosuite|numpy) " | head -6
ls /mnt/afs/260010165 2>&1 | head -5
echo PROBE_END
