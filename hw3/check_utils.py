import os
from IPython.core.debugger import set_trace

# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

def check_tensors_gpu_ready(*tensors):
    """检查所有张量是否在GPU上并且是连续的"""
    for t in tensors:
        assert t.is_contiguous, "A tensor is not contiguous"  # 断言张量是连续的
        if not os.environ.get('TRITON_INTERPRET') == '1': assert t.is_cuda, "A tensor is not on cuda"  # 如果不是模拟模式，断言张量在GPU上

def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """测试pid条件是否满足
    例如:
        '=0'  检查pid_0 == 0
        ',>1' 检查pid_1 > 1
        '>1,=0' 检查pid_0 > 1 且 pid_1 == 0
    """
    pids = pid_0[0], pid_1[0], pid_2[0]  # 获取pid值
    conds = conds.replace(' ','').split(',')  # 去除空格并分割条件
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond=='': continue  # 如果条件为空，跳过
        op, threshold = cond[0], int(cond[1:])  # 获取操作符和阈值
        if op not in ['<','>','>=','<=','=', '!=']: raise ValueError(f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule: '{condition}'.")  # 检查操作符是否合法
        op = '==' if op == '=' else op  # 将'='替换为'=='
        if not eval(f'{pid} {op} {threshold}'): return False  # 评估条件是否满足
    return True

assert test_pid_conds('')  # 测试空条件
assert test_pid_conds('>0', [1], [1])  # 测试pid_0 > 0
assert not test_pid_conds('>0', [0], [1])  # 测试pid_0 > 0不满足
assert test_pid_conds('=0,=1', [0], [1], [0])  # 测试pid_0 = 0 且 pid_1 = 1

def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """如果任何pid条件满足，停止kernel"""
    if test_pid_conds(conds, pid_0, pid_1, pid_2): set_trace()  # 如果条件满足，设置断点

def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """如果任何pid条件满足，打印txt"""
    if test_pid_conds(conds, pid_0, pid_1, pid_2): print(txt)  # 如果条件满足，打印文本

def cdiv(a,b): 
    """计算a除以b的上限值"""
    return (a + b - 1) // b  # 计算a除以b的上限值
assert cdiv(10,2)==5  # 测试cdiv函数
assert cdiv(10,3)==4  # 测试cdiv函数
import triton
print(triton.__version__)