import numpy as np
import os

def get_freer_gpu(n=1):
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    if n == 1:
        return np.argsort(memory_available)[-1]
    return np.argsort(memory_available)[-n:][::-1]
