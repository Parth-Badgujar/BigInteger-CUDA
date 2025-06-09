#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuComplex.h>
#include <math_constants.h>
#include <chrono>

uint cdiv(uint x, uint y){
    return (x+y-1)/y;
}



__device__ int get_idx(int idx, int log){
    int rev_idx = 0; 
    for(int i = 0 ; i < log; i++){
        rev_idx = (rev_idx << 1) | (idx & 1);
        idx >>= 1;
    }
    return rev_idx;
}


__global__ void _multiply_fft_kernel(uint16_t *num1, uint16_t *num2, uint16_t *num3, uint64_t root, uint64_t modulo){
    int tid = threadIdx.x; 
    __shared__ cuDoubleComplex X[64+1]; //remove bank conflicts
    __shared__ cuDoubleComplex Y[64]; 
    __shared__ cuDoubleComplex Z[64]; 
    
    X[tid] = make_cuDoubleComplex(num1[tid], 0.0); 
    Y[tid] = make_cuDoubleComplex(num2[tid], 0.0); 
    __syncthreads(); 

    int xchg_idx = get_idx(tid, 6); // log2(64)

    cuDoubleComplex temp; 
    temp = X[xchg_idx]; 
    X[xchg_idx] = X[tid]; 
    X[tid] = temp; 

    temp = Y[xchg_idx]; 
    Y[xchg_idx] = Y[tid]; 
    Y[tid] = temp; 

    __syncthreads(); 

    for (int stage = 1; stage <= 6; stage++) {
        int  length = 1 << stage;
        int  half   = length >> 1;
        int block = tid / half;
        int j     = tid % half;
        int pos   = block * length + j;
        float angle = (2.0f * CUDART_PI_F) * (float(j) / float(length));
        cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));

        cuDoubleComplex u = X[pos];
        cuDoubleComplex v = cuCmul(X[pos + half], w);
        __syncthreads();
        X[pos]        = cuCadd(u, v);
        X[pos + half] = cuCsub(u, v);

        u = Y[pos];
        v = cuCmul(Y[pos + half], w);
        __syncthreads();
        Y[pos]        = cuCadd(u, v);
        Y[pos + half] = cuCsub(u, v);
        __syncthreads();
    }

    Z[tid] = cuCmul(X[tid], Y[tid]); 
    Z[tid + 32] = cuCmul(X[tid + 32], Y[tid + 32]);

    __syncthreads(); 
    temp = Z[xchg_idx]; 
    Z[xchg_idx] = Z[tid]; 
    Z[tid] = temp; 
    __syncthreads(); 


    for (int stage = 1; stage <= 6; stage++) {
        int  length = 1 << stage;
        int  half   = length >> 1;
        int block = tid / half;
        int j     = tid % half;
        int pos   = block * length + j;
        float angle = (2.0f * CUDART_PI_F) * (float(j) / float(length));
        cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));

        cuDoubleComplex u = Z[pos];
        cuDoubleComplex v = cuCmul(Z[pos + half], w);
        __syncthreads();
        Z[pos]        = cuCadd(u, v);
        Z[pos + half] = cuCsub(u, v);
        __syncthreads();

        if((tid == 0) and stage == 3){
        for(int i = 0 ; i < 64; i++){
            printf("num %d real %f imag %f\n", i, cuCreal(Z[i]), cuCimag(Z[i]));
        }
        __syncthreads();
        }
    }

    float invN = 1.0f / float(64); 
    Z[tid].x = Z[tid].x * (invN); 
    Z[tid].y = Z[tid].y * (invN);
    Z[tid+32].x = Z[tid+32].x * (invN); 
    Z[tid+32].y = Z[tid+32].y * (invN); 
    
    
}


int main(){
    
    std::vector<uint16_t> num1 = {50972,
                                    7281,
                                    29127,
                                    50972,
                                    7281,
                                    29127,
                                    50972,
                                    7281,
                                    29127,
                                    11036,
                                    16261,
                                    16802,
                                    48157,
                                    52780,
                                    50014,
                                    58041,
                                    10593,
                                    28656,
                                    64028,
                                    54512,
                                    19318,
                                    53458,
                                    63640,
                                    45834,
                                    15101,
                                    8732,
                                    18738,
                                    62235,
                                    13550,
                                    22513,
                                    15723,
                                    217}; 
    std::vector<uint16_t> num2 = {  21845,
                                    21845,
                                    21845,
                                    21845,
                                    21845,
                                    21845,
                                    21845,
                                    21845,
                                    21845,
                                    57429,
                                    44963,
                                    61753,
                                    36117,
                                    6817,
                                    21127,
                                    27147,
                                    7945,
                                    21492,
                                    48021,
                                    8116,
                                    47257,
                                    40093,
                                    14962,
                                    17992,
                                    11326,
                                    39317,
                                    30437,
                                    13908,
                                    59315,
                                    33268,
                                    60944,
                                    162}; 
    std::vector<uint16_t> num3;
    num3.reserve(64); 

    
    uint16_t *d_num1, *d_num2, *d_num3; 
    cudaMalloc(&d_num1, 32 * sizeof(uint16_t)); 
    cudaMalloc(&d_num2, 32 * sizeof(uint16_t)); 
    cudaMalloc(&d_num3, 64 * sizeof(uint16_t)); 

    cudaMemcpy(d_num1, num1.data(), 32 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num2, num2.data(), 32 * sizeof(uint16_t), cudaMemcpyHostToDevice);


    _multiply_fft_kernel<<<1, 32>>>(d_num1, d_num2, d_num3, 3629431144387 ,6597069766657); 
    cudaDeviceSynchronize(); 
    cudaMemcpy(num3.data(), d_num3, 64 * sizeof(uint16_t), cudaMemcpyDeviceToHost); 
    
    // for(int i = 0 ; i < 64; i++){
    //     printf("num %d : %hu\n", i+1, num3[i]); 
    // }
    // printf("\n"); 
    // std::cout << "\n"; 

    return 0;
}

