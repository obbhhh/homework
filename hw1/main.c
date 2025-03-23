#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern int int_add(int a, int b);
extern float float_add(float a, float b);
extern void simd_add(float* result, float* a, float* b, int n);

void normal_add(float* result, float* a, float* b, int n){
    for(int i = 0; i < n; ++i){
        result[i] = a[i] + b[i];
    }
    return;
}

int main(){
    int len = 100000;
    float a[100000];
    float b[100000];
    float result[len];
    int add_1 = 2;
    int add_2 = 3;
    int add_res;
    float f_add_1 = 2.0;
    float f_add_2 = 3.0;
    float f_add_res;

    srand(time(NULL));
    for(int i = 0; i < len; ++i){
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    add_res = int_add(add_1, add_2);
    printf("int_add: %d + %d = %d\n", add_1, add_2, add_res);

    f_add_res = float_add(f_add_1, f_add_2);
    printf("float_add: %f + %f = %f\n", f_add_1, f_add_2, f_add_res);

    clock_t start = clock();
    simd_add(result, a, b, len);
    clock_t end = clock();
    printf("simd_add cost time: %f\n", (double)(end - start));

    start = clock();
    normal_add(result, a, b, len);
    end = clock();
    printf("normal add cost time: %f\n", (double)(end - start));

    return 0;


}