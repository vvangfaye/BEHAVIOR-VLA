"""Hold real device memory to test a policy within a smaller free-memory budget.

This is a constrained test on the actual device, not proof of a different GPU's
physical architecture. The reservation stays allocated throughout inference.
"""
import ctypes

class DeviceMemoryBudget:
    def __init__(self, budget_mib):
        self.budget_bytes = int(budget_mib) * 1024**2
        self.lib = ctypes.CDLL('libcuda.so.1')
        self.context = ctypes.c_void_p()
        self.pointer = ctypes.c_uint64()
        self.held_bytes = 0
        self.proof = {}

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result:
            raise RuntimeError('CUDA driver %s failed (%s)' % (name, result))

    def enter(self):
        self.call('cuInit', 0)
        device = ctypes.c_int()
        self.call('cuDeviceGet', ctypes.byref(device), 0)
        self.call('cuCtxCreate_v2', ctypes.byref(self.context), 0, device)
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.call('cuMemGetInfo_v2', ctypes.byref(free), ctypes.byref(total))
        if not 0 < self.budget_bytes < free.value:
            raise RuntimeError('Requested memory budget is not below available device memory')
        self.held_bytes = free.value - self.budget_bytes
        self.call('cuMemAlloc_v2', ctypes.byref(self.pointer), ctypes.c_size_t(self.held_bytes))
        self.call('cuMemGetInfo_v2', ctypes.byref(free), ctypes.byref(total))
        requested = free.value + 64 * 1024**2
        calibration = ctypes.c_uint64()
        code = self.lib.cuMemAlloc_v2(ctypes.byref(calibration), ctypes.c_size_t(requested))
        if code == 0:
            self.call('cuMemFree_v2', calibration)
            raise RuntimeError('Over-budget CUDA allocation unexpectedly succeeded')
        if code != 2:
            raise RuntimeError('Calibration did not produce CUDA_ERROR_OUT_OF_MEMORY')
        self.proof = dict(method='live_device_memory_reservation', physical_device_unchanged=True,
                          requested_budget_bytes=self.budget_bytes, reserved_bytes=self.held_bytes,
                          observed_free_bytes=free.value, physical_total_bytes=total.value,
                          over_budget_test_bytes=requested, over_budget_cuda_error=code,
                          hard_allocation_limit_calibrated=True)
        return self.proof

    def close(self):
        if self.pointer.value:
            self.call('cuMemFree_v2', self.pointer)
            self.pointer.value = 0
        if self.context.value:
            self.call('cuCtxDestroy_v2', self.context)
            self.context.value = None
