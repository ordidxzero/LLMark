from LLMark.llmark.sqlite.mapping import KernelOpStateMachine
import unittest
from pathlib import Path

class StateMachineTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_1(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn_end')
        sm.set_kernel_id(4, 'o_proj')
        sm.set_kernel_id(5, 'gate_up_proj')
        sm.set_kernel_id(6, 'down_proj')

        expected = ['qkv_proj', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'down_proj']

        actual = [sm.get_name(i+1) for i in range(len(expected))]

        self.assertEqual(actual, expected)
    
    def test_2(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn_end')
        sm.set_kernel_id(4, 'o_proj')
        sm.set_kernel_id(5, 'gate_up_proj')
        sm.set_kernel_id(6, 'down_proj')

        expected = ['qkv_proj', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'gate_up_proj']

        actual = [sm.get_name(i) for i in [1, 2, 3, 4, 5, 5]]

        self.assertEqual(actual, expected)
    
    def test_3(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn_end')
        sm.set_kernel_id(1, 'o_proj')
        sm.set_kernel_id(4, 'gate_up_proj')
        sm.set_kernel_id(5, 'down_proj')
        sm.set_kernel_id(6, 'lm_head')

        expected = ['qkv_proj', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'down_proj', 'lm_head']

        actual = [sm.get_name(i) for i in [1, 2, 3, 1, 4, 5, 6]]

        self.assertEqual(actual, expected)
    
    def test_4(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn_end')
        sm.set_kernel_id(1, 'o_proj')
        sm.set_kernel_id(4, 'gate_up_proj')
        sm.set_kernel_id(5, 'down_proj')
        sm.set_kernel_id(6, 'lm_head')

        expected = ['qkv_proj', 'o_proj', 'qkv_proj', 'o_proj', 'qkv_proj']

        actual = [sm.get_name(i) for i in [1, 1, 1, 1, 1]]

        self.assertEqual(actual, expected)

    def test_5(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn_end')
        sm.set_kernel_id(1, 'o_proj')
        sm.set_kernel_id(4, 'gate_up_proj')
        sm.set_kernel_id(4, 'down_proj')
        sm.set_kernel_id(5, 'lm_head')

        expected = ['qkv_proj', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'down_proj', 'lm_head']

        actual = [sm.get_name(i) for i in [1, 2, 3, 1, 4, 4, 5]]

        self.assertEqual(actual, expected)

    def test_6(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn_end')
        sm.set_kernel_id(4, 'o_proj')
        sm.set_kernel_id(5, 'gate_up_proj')
        sm.set_kernel_id(6, 'down_proj')
        sm.set_kernel_id(6, 'lm_head')

        expected = ['qkv_proj', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'down_proj'] * 32
        expected += ['lm_head']

        a = [1,2,3,4,5,6] * 32
        a += [6]

        actual = [sm.get_name(i) for i in a]

        self.assertEqual(actual, expected)

    def test_7(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn_end')
        sm.set_kernel_id(1, 'o_proj')
        sm.set_kernel_id(4, 'gate_up_proj')
        sm.set_kernel_id(5, 'down_proj')
        sm.set_kernel_id(4, 'lm_head')

        expected = ['qkv_proj', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'down_proj'] * 32
        expected += ['lm_head']

        a = [1,2,3,1,4,5] * 32
        a += [4]

        actual = [sm.get_name(i) for i in a]

        self.assertEqual(actual, expected)

    def test_8(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn')
        sm.set_kernel_id(4, 'self_attn_end')
        sm.set_kernel_id(1, 'o_proj')
        sm.set_kernel_id(5, 'gate_up_proj')
        sm.set_kernel_id(6, 'down_proj')
        sm.set_kernel_id(5, 'lm_head')

        expected = ['qkv_proj', 'self_attn', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'down_proj'] * 32
        expected += ['lm_head']

        a = [1,2,3,4,1,5,6] * 32
        a += [5]

        actual = [sm.get_name(i) for i in a]

        self.assertEqual(actual, expected)

    def test_8(self):
        sm = KernelOpStateMachine()
        
        sm.set_kernel_id(1, 'qkv_proj')
        sm.set_kernel_id(2, 'self_attn')
        sm.set_kernel_id(3, 'self_attn')
        sm.set_kernel_id(4, 'self_attn_end')
        sm.set_kernel_id(1, 'o_proj')
        sm.set_kernel_id(5, 'gate_up_proj')
        sm.set_kernel_id(6, 'down_proj')
        sm.set_kernel_id(1, 'lm_head')

        expected = ['qkv_proj', 'self_attn', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'down_proj'] * 32
        expected += ['lm_head']
        expected += ['qkv_proj', 'self_attn', 'self_attn', 'self_attn_end', 'o_proj', 'gate_up_proj', 'down_proj'] * 32
        expected += ['lm_head']

        a = [1,2,3,4,1,5,6] * 32
        a += [1]
        a += [1,2,3,4,1,5,6] * 32
        a += [1]

        actual = [sm.get_name(i) for i in a]

        self.assertEqual(actual, expected)