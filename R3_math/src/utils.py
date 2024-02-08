import random
import numpy as np
import torch
import signal
import json

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def is_numeric(value):
    try:
        value = float(value)
        return True
    except Exception as e:
        return False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def write_data(file: str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


from torch.distributed import all_reduce, ReduceOp
def do_gather(var):
    var = torch.FloatTensor(var).cuda()
    all_reduce(var, op=ReduceOp.SUM)
    var = var.cpu().numpy().tolist()
    return var


import scipy.signal as scipy_signal
def discount_cumsum(rewards, discount):
    return scipy_signal.lfilter([1], [1, -discount], x=rewards[::-1])[::-1]
