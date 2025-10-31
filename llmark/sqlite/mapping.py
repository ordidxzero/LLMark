import sqlite3
from typing import List, Dict
from collections import OrderedDict, defaultdict

class KernelOpStateMachine:
    _i: int
    _state: Dict[int, List[str]]
    _tick: Dict[int, int]
    _skip: Dict[int, bool]
    def __init__(self):
        self._i = 0
        self._state = defaultdict(list)
        self._tick = defaultdict(int)
        self._skip = defaultdict(bool)
    
    def set_kernel_id(self, kernel_id:int, name: str):
        self._state[kernel_id].append(name)

    def get_name(self, kernel_id: int):
        names = self._state.get(kernel_id, None)
        if names is None:
            return None
        if 'lm_head' in names:
            if len(names) == 1:
                return 'lm_head'
            
            if len(names) == 2:
                if self._tick[kernel_id] != 0 and self._tick[kernel_id] % 32 == 0 and not self._skip[kernel_id]:
                    self._skip[kernel_id] = True
                    return 'lm_head'
                
                self._skip[kernel_id] = False
                self._tick[kernel_id] += 1
                return names[0]
            if len(names) == 3:
                if self._tick[kernel_id] != 0 and self._tick[kernel_id] % 64 == 0 and not self._skip[kernel_id]:
                    self._skip[kernel_id] = True
                    return 'lm_head'

                self._skip[kernel_id] = False
                idx = self._tick[kernel_id] % (len(names) - 1)
                self._tick[kernel_id] += 1
                return names[idx]
        else:
            if len(names) == 1:
                return names[0]
            
            idx = self._tick[kernel_id] % len(names)
            self._tick[kernel_id] += 1
            return names[idx]
        
        return None

class KernelMapping:
    qkv_proj: int
    self_attn: List[int]
    o_proj: int
    gate_up_proj: int
    down_proj: int
    lm_head: int
    _pre_defined_kernel: OrderedDict[str, List[str]]
    _cursor: sqlite3.Cursor
    _state: KernelOpStateMachine
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor
        self.self_attn = []
        self.state = KernelOpStateMachine()
        
        for kernel_name, nvtx_name in self._pre_defined_kernel.items():
            QUERY = f"SELECT id FROM StringIds WHERE value LIKE '%{kernel_name}%'"
            self._cursor.execute(QUERY)
            kernel_id = self._cursor.fetchone()
            assert kernel_id is not None, f"Mapping Initialization Error"
            kernel_id = kernel_id[0]
            for name in nvtx_name:
                self.state.set_kernel_id(kernel_id, name)
                if 'self_attn' in name:
                    self.self_attn.append(kernel_id)
                else:
                    setattr(self, name, kernel_id)

    def get_name(self, kernel_id: int):
        return self.state.get_name(kernel_id)