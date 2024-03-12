import cuda
import cudart
import nvrtc


def check_error(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f'Cuda Error: {err}')
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaSuccess:
            raise RuntimeError(f'Cuda Error: {err}')
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f'Nvrtc Error: {err}')
    else:
        raise RuntimeError(f'Unknown error type: {err}')


def get_sm_version():
    check_error(cuda.cuInit(0))

    check_error(cuda.cuDeviceGet(0, cuda.CUdevice()))

    sm_major = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    sm_minor = cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

    check_error(cuda.cuDeviceGetAttribute(
        sm_major, cuda.CUdevice(), cudart.byref(cudart.CUjit_option())))
    check_error(cuda.cuDeviceGetAttribute(
        sm_minor, cuda.CUdevice(), cudart.byref(cudart.CUjit_option())))

    return cudart.CUjit_option().value / 10, cudart.CUjit_option().value % 10

