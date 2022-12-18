import glob
import platform
import os
import sys
import subprocess
from pathlib import Path
import paddle
import cv2


IS_WINDOWS = platform.system() == 'Windows'


def _find_cuda_home():
    cuda_home = os.environ.get('CUDA_HOME') or \
        os.environ.get('CUDA_PATH')
    
    if cuda_home is not None:
        return cuda_home
    
    try:
        which = 'where' if IS_WINDOWS else 'which'
        nvcc = subprocess.check_output([
            which, 'nvcc'
        ]).decode('utf-8').rstrip('\r\n', '')
        cuda_home = Path(nvcc).parents[1]
        return cuda_home
    except: pass

    if IS_WINDOWS:
        cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
        if len(cuda_homes) == 0:
            cuda_home = ''
        else:
            cuda_home = cuda_homes[0]
    else:
        cuda_home = '/usr/local/cuda'
    cuda_home = cuda_home if os.path.exists(cuda_home) else None
    return cuda_home


def _get_nvcc_info(cuda_home):
    if cuda_home is None or not os.path.exists(cuda_home):
        return 'Not Available'
    
    try:
        nvcc = os.path.join(cuda_home, 'bin/nvcc')
        if not IS_WINDOWS:
            nvcc = subprocess.check_output(
                f'{nvcc} -V', shell=True
            )
        else:
            nvcc = subprocess.check_output(
                    f"\"{nvcc}\" -V", shell=True)
        nvcc = nvcc.decode().strip().split('\n')[-1]
        return nvcc
    except subprocess.SubprocessError:
        return 'Not Available'


def _get_gpu_info():
    try:
        gpu_info = subprocess.check_output(
            ['nvidia-smi', '-L']).decode().strip()
        gpu_infos = gpu_info.split('\n')
        return gpu_infos
    except:
        return ' Can not get GPU information. Please make sure CUDA have been installed successfully.'


def get_sys_env():
    env_info = {}
    env_info['platform'] = platform.platform()
    env_info['Python'] = sys.version.replace('\r', '')
    try:
        gcc = subprocess.check_output(['gcc', '--version']).decode()
        env_info['GCC'] = gcc.strip().split('\n')[0]
    except: pass

    if paddle.is_compiled_with_cuda():
        cuda_home = _find_cuda_home()
        env_info['NVCC'] = _get_nvcc_info(cuda_home)
        v = paddle.get_cudnn_version()
        v = str(v // 1000) + '.' + str(v%1000 // 100)
        env_info['cudnn'] = v
        if 'gpu' in paddle.get_device():
            gpu_nums = paddle.distributed.ParallelEnv().nranks
        else:
            gpu_nums = 0
        env_info['GPUs used'] = gpu_nums
        env_info['GPU'] = _get_gpu_info()
        env_info['PaddlePaddle'] = paddle.__version__
        env_info['OpenCV'] = cv2.__version__
    return env_info


if __name__ == '__main__':
    info = get_sys_env()
    for name, val in info.items():
        print(name, val)