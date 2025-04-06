// driver_api.cu
#include <cuda.h>
#include <stdio.h>
#include <cerrno>

int funcPrepare(const char* filePath, CUmodule& module, CUfunction& function, const char* funcName, char*& cubin, CUresult& err){
    // 加载CUBIN文件
    // 修改文件加载部分的代码
    FILE* file = fopen(filePath, "rb");
    if (!file) {  // 必须检查文件是否成功打开
        printf("Error: Cannot open cubin_data.cubin Reason: %s\n", strerror(errno));
        printf("Current working directory: ");
        system("pwd");  // 打印当前工作目录
        return -1;
    }

    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);
    cubin = (char*)malloc(size);
    size_t read_size = fread(cubin, 1, size, file);
    if (read_size != size) {
        printf("Error: Failed to read full file (expected %zu, got %zu)\n", size, read_size);
        return -1;
    }
    fclose(file);

    // 加载模块和函数
    if ((err = cuModuleLoadData(&module, cubin)) != CUDA_SUCCESS) {
        printf("cuModuleLoadData failed: %d\n", err);
        return -1;
    }
    if ((err = cuModuleGetFunction(&function, module, funcName)) != CUDA_SUCCESS) {
        printf("cuModuleGetFunction failed: %d \n %s \n", err, funcName);
        return -1;
    }
     return 0;  // Return success
}

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module_add;
    CUmodule module_matmul;
    CUfunction function_add;
    CUfunction function_matmul;
    char * cubin;
    CUresult err;

    // 初始化Driver API
    if ((err = cuInit(0)) != CUDA_SUCCESS) {
        printf("cuInit failed: %d\n", err);
        return -1;
    }
    if ((err = cuDeviceGet(&device, 0)) != CUDA_SUCCESS) {
        printf("cuDeviceGet failed: %d\n", err);
        return -1;
    }
    if ((err = cuCtxCreate(&context, 0, device)) != CUDA_SUCCESS) {
        printf("cuCtxCreate failed: %d\n", err);
        return -1;
    }

    // 加载add_kernel
    if (funcPrepare("cubin_data.cubin", module_add, function_add, "add_kernel", cubin, err) < 0) {
        return -1;
    }

    // 加载naive_matmul_k kernel
    if (funcPrepare("naive_matmul_k.cubin", module_matmul, function_matmul, "naive_matmul_k", cubin, err) < 0) {
        return -1;
    }

    // 准备数据
    CUdeviceptr d_x, d_y, d_output;
    cuMemAlloc_v2(&d_x, 1024 * sizeof(float));
    cuMemAlloc_v2(&d_y, 1024 * sizeof(float));
    cuMemAlloc_v2(&d_output, 1024 * sizeof(float));

    // 准备输入数据
    float h_x[1024], h_y[1024], h_output[1024];
    for (int i = 0; i < 1024; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }
    cuMemcpyHtoD_v2(d_x, h_x, 1024 * sizeof(float));
    cuMemcpyHtoD_v2(d_y, h_y, 1024 * sizeof(float));

    // 启动内核
    int len = 1024;
    void* args[] = { &d_x, &d_y, &d_output, &len };
    cuLaunchKernel(function_add, 1, 1, 1, 1024, 1, 1, 0, NULL, args, NULL);

    // 验证结果
    cuMemcpyDtoH_v2(h_output, d_output, 1024 * sizeof(float));
    bool passed = true;
    for (int i = 0; i < 1024; i++) {
        if (fabs(h_output[i] - 3.0f) > 1e-6) {  // 1 + 2 = 3
            printf("Verification failed at index %d: expected 3.0, got %f\n", i, h_output[i]);
            passed = false;
            break;
        }
    }
    if (passed) {
        printf("Kernel verification PASSED!\n");
    }

    // ====================== 测试 naive_matmul_k ======================
    // 准备矩阵乘法测试数据 (128x256) * (256x512) = (128x512)
    int M = 128, N = 512, K = 256;
    CUdeviceptr d_A, d_B, d_C;
    cuMemAlloc_v2(&d_A, M * K * sizeof(float));
    cuMemAlloc_v2(&d_B, K * N * sizeof(float));
    cuMemAlloc_v2(&d_C, M * N * sizeof(float));

    // 初始化随机输入数据
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    // 保持与Python相同的随机数范围(0.1倍缩放)
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX * 0.1f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX * 0.1f;
    
    cuMemcpyHtoD_v2(d_A, h_A, M * K * sizeof(float));
    cuMemcpyHtoD_v2(d_B, h_B, K * N * sizeof(float));

    // 定义block size常量
    const int bs = 16;
    int stride_am = K, stride_ak = 1;
    int stride_bk = N, stride_bn = 1;
    int stride_cm = N, stride_cn = 1;
    int bm = bs, bn = bs, bk = bs;

    // 设置kernel参数（正确传递设备指针）
    void* matmul_args[] = {
        &d_A,      // 直接传递设备指针值
        &d_B, 
        &d_C,
        &M, &N, &K,
        &stride_am, &stride_ak,
        &stride_bk, &stride_bn,
        &stride_cm, &stride_cn,
        &bm, &bn, &bk
    };

    // 启动kernel (使用正确的block配置)
    dim3 grid((M + bm - 1) / bm, (N + bn - 1) / bn);
    dim3 block(16, 16);  // 必须与Triton kernel的bm/bn一致
    // 添加共享内存配置（根据Triton kernel需求）
    const int shared_mem = 2 * bm * bk * sizeof(float) + 2 * bk * bn * sizeof(float);
    
    CUstream stream;
    cuStreamCreate(&stream, 0);
    
    cuLaunchKernel(function_matmul, 
                  grid.x, grid.y, 1,    // grid dim
                  block.x, block.y, 1,  // block dim
                  shared_mem, stream, matmul_args, NULL);  // 添加共享内存配置
    
    // 等待kernel完成
    cuStreamSynchronize(stream);
    cuStreamDestroy(stream);

    // 验证结果
    cuMemcpyDtoH_v2(h_C, d_C, M * N * sizeof(float));
    
    // 简单验证几个样本点
    bool matmul_passed = true;
    for (int i = 0; i < 10; i++) {
        int row = rand() % M;
        int col = rand() % N;
        float expected = 0.0f;
        for (int k = 0; k < K; k++) {
            expected += h_A[row * K + k] * h_B[k * N + col];
        }
        if (fabs(h_C[row * N + col] - expected) > 1e-3) {
            printf("Matmul verification failed at (%d,%d): expected %.4f, got %.4f, i=%d\n", 
                  row, col, expected, h_C[row * N + col], i);
            matmul_passed = false;
            break;
        }
    }
    if (matmul_passed) {
        printf("Matmul kernel verification PASSED!\n");
    }

    // 清理资源
    free(h_A); free(h_B); free(h_C);
    cuMemFree_v2(d_A);
    cuMemFree_v2(d_B);
    cuMemFree_v2(d_C);
    cuMemFree_v2(d_x);
    cuMemFree_v2(d_y);
    cuMemFree_v2(d_output);
    cuModuleUnload(module_add);
    cuModuleUnload(module_matmul);
    cuCtxDestroy(context);
    free(cubin);
    return 0;
}