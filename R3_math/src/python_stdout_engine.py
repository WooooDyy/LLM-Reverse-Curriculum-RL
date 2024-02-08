
from io import StringIO
from contextlib import redirect_stdout
import multiprocessing
import threading
import re
import math

def floatify(num: str):
    try:
        num = float(num)
        if num.is_integer():
            return round(num)
        else:
            return num
    except Exception:
        return None

def number_it(num: str):
    if 'frac' in num:
        pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
        num = re.sub(pattern, r"\1/\2", num)
        try:
            num = str(eval(num))
        except Exception:
            pass
    elif ',' in num:
        num = num.replace(',', '')

    if floatify(num) is not None:
        return floatify(num)
    else:
        try:
            num = eval(num)
            if isinstance(num, list) or isinstance(num, tuple):
                num = num[0]
            if floatify(num) is not None:
                return floatify(num)
            else:
                return None
        except Exception:
            return None

def format_code(code_str: str):
    code = 'def run_it():\n'
    for line in code_str.split('\n'):
        code += '  ' + line + '\n'
    code += 'run_it()'
    return code

def compare_both_string_and_number_format(answer, groundtruth_str, groundtruth_num):
    if answer == groundtruth_str:
        return True
    else:
        if groundtruth_num is not None and number_it(answer) is not None:
            if compare_two_numbers(number_it(answer), groundtruth_num):
                return True
            else:
                return False
        else:
            return False

def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        if isinstance(gt, int):
            return round(p) == gt
        else:
            return within_eps(pred=p, gt=gt)
    except Exception:
        return False

def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.04
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False

class CodeExecutor:
    def __init__(self, code, timeout, use_process: bool):
        self.code = format_code(code)
        self.timeout = timeout
        self.error = ''
        self.use_process = use_process

    def execute_code(self, return_val):
        try:
            f = StringIO()
            with redirect_stdout(f):
                exec(self.code, {})
            s = f.getvalue()
            s = s.strip('\n')
            return_val['result'] = s
        except Exception:
            pass

    def run(self):
        if self.use_process:
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            process = multiprocessing.Process(
                target=self.execute_code, args=(return_dict,))
            process.start()
            process.join(timeout=self.timeout)
            process.terminate()
        else:
            return_dict = {}
            thread = threading.Thread(
                target=self.execute_code, args=(return_dict,))
            thread.start()
            thread.join(timeout=self.timeout)
            if thread.is_alive():
                thread.join()  # Ensures the thread is terminated before continuing
                print('time out!')
                self.error = 'Execution timed out'

        if 'result' in return_dict:
            return return_dict['result']
        else:
            return ''


def run_python_stdout_code(code_gen, timeout: int=5, use_process: bool = True):
    executor = CodeExecutor(code_gen, timeout, use_process)
    s = executor.run()
    if number_it(s) is not None:
        return str(number_it(s))
    return s

