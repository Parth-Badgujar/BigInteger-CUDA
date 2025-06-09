#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include "bigint.cuh"
    

//512 bit numbers
const char* num1 = "44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444"; 
const char* num2 = "33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333"; 

typedef std::vector<std::vector<BigInt>> Matrix; 



void benchmark_cuda_matmul(int n, int m, int k, int iters){
    uint16_t* mat1 = new uint16_t[n * m * 32];
    uint16_t* mat2 = new uint16_t[m * k * 32];
    uint16_t* mat3 = new uint16_t[n * k * 64];
    BigInt x(num1); 
    BigInt y(num2); 

    std::vector<uint16_t> l1 = x.get_limbs(); 
    std::vector<uint16_t> l2 = y.get_limbs(); 

    //matrix filled with num1
    for (int i = 0; i < n * m; i++){
        for(int j = 0; j < 32; j++){
            mat1[i * 32 + j] = l1[j]; 
        }
    }

    //matrix filled with num2
    for (int i = 0; i < k * m; i++){
        for(int j = 0; j < 32; j++){
            mat2[i * 32 + j] = l2[j]; 
        }
    }

    uint16_t *d_mat1, *d_mat2, *d_mat3;
    CUDA_CHECK(cudaMalloc((void**)&d_mat1, n * m * 32 * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_mat2, m * k * 32 * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_mat3, n * k * 64 * sizeof(uint16_t)));

    CUDA_CHECK(cudaMemcpy(d_mat1, mat1, n * m * 32 * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mat2, mat2, m * k * 32 * sizeof(uint16_t), cudaMemcpyHostToDevice));

    //wrampup
    for(int i = 0 ; i < 2; i++){
        matmul(d_mat1, d_mat2, d_mat3, n, m, k);
    }

    CUDAtimer t; 
    t.start_timer(); 
    for(int i = 0 ; i < iters; i++){
        matmul(d_mat1, d_mat2, d_mat3, n, m, k);
    }
    t.stop_timer(); 
    float ms = t.get_time() / iters; 
    std::cout << "512 bit " << "n=" << n << " " 
                << "m=" << m << " "
                << "k=" << k << " "
                << "time taken : " << ms << "ms" << "\n"; 

    CUDA_CHECK(cudaMemcpy(mat3, d_mat3, n * k * 64 * sizeof(uint16_t), cudaMemcpyDeviceToHost));   

    CUDA_CHECK(cudaFree(d_mat1)); 
    CUDA_CHECK(cudaFree(d_mat2)); 
    CUDA_CHECK(cudaFree(d_mat3));

    delete mat1; 
    delete mat2; 
    delete mat3; 
}


int main(){
    for(int i = 16 ; i < 512; i*=2){
        benchmark_cuda_matmul(i, i, i, 10); 
    }
    BigInt x(num1); 
    BigInt y(num2); 
    BigInt z = x + y; 
    BigInt prod = x * y; //fft 
    BigInt sim_prod = multiply_simple(x, y);
}