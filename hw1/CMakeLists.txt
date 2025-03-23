cmake_minimum_required(VERSION 3.10)

project(hw1 C ASM_NASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -fno-tree-vectorize")


add_executable(hw1 main.c simd_add.asm int_add.asm float_add.asm)

