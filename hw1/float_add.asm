section .text
global float_add

; 函数原型：
; float float_add(float a, float b);
;
; 参数：
;   xmm0: a (第一个浮点数)
;   xmm1: b (第二个浮点数)
; 返回值：
;   xmm0: 结果

float_add:
    ; 将参数 a 和 b 相加
    addss xmm0, xmm1    ; 将 xmm1 加到 xmm0
    ret                 ; 返回结果