#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>  // 引入 SIMD 头文件

int main() {
    int len = 10240;
    float* a = (float*)aligned_alloc(16, len * sizeof(float));  // 16 字节对齐
    float* b = (float*)aligned_alloc(16, len * sizeof(float));  // 16 字节对齐
    float* result = (float*)aligned_alloc(16, len * sizeof(float));  // 16 字节对齐

    srand(time(NULL));
    for (int i = 0; i < len; ++i) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // SIMD 加法
    clock_t start = clock();
    for (int i = 0; i < len; i += 4) {
        __m128 vec1 = _mm_load_ps(&a[i]);  // 加载 4 个浮点数
        __m128 vec2 = _mm_load_ps(&b[i]);
        __m128 res = _mm_add_ps(vec1, vec2);  // SIMD 加法
        _mm_store_ps(&result[i], res);  // 存储结果
    }
    clock_t end = clock();
    printf("simd_add cost time: %f\n", (double)(end - start));

    // 逐个加法
    start = clock();
    for (int i = 0; i < len; ++i) {
        result[i] = a[i] + b[i];
    }
    end = clock();
    printf("normal add cost time: %f\n", (double)(end - start));

    // 释放内存
    free(a);
    free(b);
    free(result);

    return 0;
}
