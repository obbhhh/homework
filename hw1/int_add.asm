section .text
global int_add

; 函数原型：
; int int_add(int a, int b);
;
; 参数：
;   rdi: a (第一个整数)
;   rsi: b (第二个整数)
; 返回值：
;   rax: 结果

int_add:
    ; 将参数 a 和 b 相加
    mov eax, edi        ; 将第一个参数 a 加载到 eax
    add eax, esi        ; 将第二个参数 b 加到 eax
    ret                 ; 返回结果