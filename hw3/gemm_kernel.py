from check_utils import check_tensors_gpu_ready

import triton
from triton import language as tl
import torch
import inspect
from triton.compiler.compiler import CompiledKernel


@triton.jit
def get_1d_offset(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1): 
    # 使用 tl.expand_dims 将 offs_0 和 offs_1 转换为二维张量
    # tl.expand_dims(offs_0, 1) 将 offs_0 转换为 (offs_0, 1) 形状的张量
    # tl.expand_dims(offs_1, 0) 将 offs_1 转换为 (1, offs_1) 形状的张量
    return tl.expand_dims(offs_0, 1)*stride_0 + tl.expand_dims(offs_1, 0)*stride_1

@triton.jit
def get_1d_mask(offs, max):
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    # 使用 tl.expand_dims 将 offs_0 和 offs_1 转换为二维张量
    # tl.expand_dims(offs_0, 1) 将 offs_0 转换为 (offs_0, 1) 形状的张量
    # tl.expand_dims(offs_1, 0) 将 offs_1 转换为 (1, offs_1) 形状的张量
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)

@triton.jit
def naive_matmul_k(
    a_ptr, b_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr
):
    # 获取当前线程块的 ID
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    # 沿 m/n/k 维度分割计算
    rm = get_1d_offset(size=bm, n_prev_chunks=pid_m)  # 计算 m 维度的偏移量
    rn = get_1d_offset(size=bn, n_prev_chunks=pid_n)  # 计算 n 维度的偏移量
    rk = get_1d_offset(size=bk, n_prev_chunks=0)  # 计算 k 维度的偏移量
    # 计算 a 和 b 的相关偏移量
    offs_a = a_ptr + get_2d_offset(rm, rk, stride_am, stride_ak)  # 计算 a 的偏移量
    offs_b = b_ptr + get_2d_offset(rk, rn, stride_bk, stride_bn)  # 计算 b 的偏移量
    # 初始化并迭代更新累加器
    acc = tl.zeros((bm, bn), dtype=tl.float32)  # 初始化累加器
    for _ in range(0, k, bk):
        # todo umer: 加载 a 和 b 时是否需要掩码？
        a = tl.load(offs_a)  # 加载 a 的数据
        b = tl.load(offs_b)  # 加载 b 的数据
        acc += tl.dot(a, b, allow_tf32=False)  # 在块内进行矩阵乘法；注意：对于较旧的 GPU，allow_tf32 必须设置为 False，否则无法编译
        # 增加偏移量，以便下一次迭代加载下一个块
        offs_a += bk * stride_ak
        offs_b += bk * stride_bk
    c = c_ptr + get_2d_offset(rm, rn, stride_cm, stride_cn)  # 计算 c 的偏移量
    mask = get_2d_mask(rm, rn, m, n)  # 计算掩码
    tl.store(c, acc, mask=mask)  # 将结果存储到 c 中

from functools import partial



torch.manual_seed(0)
a = torch.randn((128, 256), device='cuda', dtype=torch.float32) * 0.1
b = torch.randn((256, 512), device='cuda', dtype=torch.float32) * 0.1
bs = 16






# 检查矩阵维度是否兼容
assert a.shape[1] == b.shape[0], "矩阵维度不兼容，无法进行矩阵乘法"
# 检查张量是否准备好在 GPU 上运行
check_tensors_gpu_ready(a, b)
# 获取矩阵 a 和 b 的形状
(m, k), (_, n) = a.shape, b.shape
# 创建一个空的输出张量 c
c = torch.empty((m, n), device=a.device, dtype=a.dtype)  # 保持与输入相同类型
# 定义网格函数，用于计算线程块的数量
grid = lambda meta: (triton.cdiv(m, meta['bm']),  triton.cdiv(n, meta['bn']))
# 处理 group_sz 参数，如果为 None，则使用空字典
group_sz = {} # 在 naive_matmul 中未使用，但在后续的 grouped_matmul 中会用到
# 调用 matmul_k_fn 函数，传入必要的参数
compiled_kernel: CompiledKernel = naive_matmul_k[grid](
    a, b, c,
    m, n, k,
    a.stride(0), a.stride(1),
    b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    bm=bs, bn=bs, bk=bs, # 注意：对于较旧的 GPU，allow_tf32 必须设置为 False，否则无法编译
    **group_sz
)
print(type(compiled_kernel))
torch_output = torch.matmul(a, b)
if torch.allclose(c, torch_output, atol=5e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

# 导出到 cubin 文件
print(compiled_kernel.asm.keys())
with open("./naive_matmul_k.cubin", "wb") as f:
    f.write(compiled_kernel.asm["cubin"])

