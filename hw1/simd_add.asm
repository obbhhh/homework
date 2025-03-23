section .text
global simd_add

; 函数原型：
; void simd_add(float* result, float* vec1, float* vec2, int length);
;
; 参数：
;   rdi: result (输出向量的指针)
;   rsi: vec1 (第一个输入向量的指针)
;   rdx: vec2 (第二个输入向量的指针)
;   rcx: length (向量长度)

simd_add:
    ; 保存寄存器状态
    push rbx
    push r12
    push r13
    push r14

    ; 初始化指针和计数器
    mov rbx, rdi          ; result 指针
    mov r12, rsi          ; vec1 指针
    mov r13, rdx          ; vec2 指针
    mov r14, rcx          ; length

    ; 检查长度是否为 0
    test r14, r14
    jz .done

    ; SIMD 加法循环
    ; 每次处理 4 个浮点数（128 位）
    align 16
.loop:
    ; 加载 vec1 和 vec2 的 4 个浮点数
    movaps xmm0, [r12]
    movaps xmm1, [r13]

    ; 相加
    addps xmm0, xmm1

    ; 存储结果到 result
    movaps [rbx], xmm0

    ; 更新指针
    add rbx, 16          ; 指向下一个 4 个浮点数的位置
    add r12, 16
    add r13, 16

    ; 减少计数器
    sub r14, 4
    jg .loop             ; 如果还有数据，继续循环

.done:
    ; 恢复寄存器状态
    pop r14
    pop r13
    pop r12
    pop rbx
    ret