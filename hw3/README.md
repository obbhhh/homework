参考资料:https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/#cubin
https://zhuanlan.zhihu.com/p/11901442836
1. 首先实现一个简单的 add_kernel.py 
2. 通过命令 `python3 ~/miniforge3/lib/python3.12/site-packages/triton/tools/compile.py ./add_kernel.py  --kernel-name add_kernel   --signature "*fp32,*fp32,*fp32,i32,64"  --grid=1024,1024,1024 ` 可以生成 triton kernel 编译后的 cubin 二进制以及封装好的利用 cuda driver 装载及调用 kernel 的 c 函数和借口
3. .py 文件中调用 triton kernel, JIT 会返回编译好的 Kernel 通过其 .asm['cubin'] 可以将其 cubin 二进制导出到文件中
4. 在 main.cu 中模仿利用 compile.py 导出的 .c 文件中的方法, 装载 triton kernel 并调用