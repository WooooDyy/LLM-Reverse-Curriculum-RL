import copy
from typing import Any, Dict

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []
    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        
        for c in self.HEADERS:
            self.exec_code(c)
        
    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)
        
    def eval_code(self, expr: str) -> Any:
        # try:
        #     return eval(expr, self._global_vars)
        # except:
        #     raise

        return eval(expr, self._global_vars)

    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']

def process_code(code_gen, truncate_first_return=False):
    snippet = code_gen.split('\n')
    ## post process the code
    updated_code_snippet = ['import math', 'import sympy']
    for snippet_line in snippet:
        if snippet_line.startswith('def solution'):
            updated_code_snippet.append(snippet_line)
            continue
        if snippet_line.strip() == "":
            break
        if truncate_first_return:
            if snippet_line == "    return result":
                break
        updated_code_snippet.append(snippet_line)
    updated_code_gen = '\n'.join(updated_code_snippet)
    return updated_code_gen

def run_python_code(code_gen: str):
    answer_expr = "solution()"
    runtime = GenericRuntime()
    updated_code_gen = process_code(code_gen)
    runtime.exec_code(updated_code_gen)
    return runtime.eval_code(answer_expr)