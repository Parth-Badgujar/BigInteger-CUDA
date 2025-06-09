#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuComplex.h>
#include <chrono>
#include "bigint.cuh"


#define BLOCK_SIZE 4

__host__ __device__ uint32_t next_power_of_2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

__device__ __constant__ uint64_t root = 3629431144387; 
__device__ __constant__ uint64_t mod = 6597069766657; 

__device__ __constant__ int idx_map[28][2] = {
    {32,  1}, {16,  2}, {48,  3}, { 8,  4},
    {40,  5}, {24,  6}, {56,  7}, {36,  9},
    {20, 10}, {52, 11}, {44, 13}, {28, 14},
    {60, 15}, {34, 17}, {50, 19}, {42, 21},
    {26, 22}, {58, 23}, {38, 25}, {54, 27},
    {46, 29}, {62, 31}, {49, 35}, {41, 37},
    {57, 39}, {53, 43}, {61, 47}, {59, 55}
};


uint cdiv(uint x, uint y){
    return (x+y-1)/y;
}

__device__ uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    uint128_t res = (uint128_t)(((uint128_t)(a) * (uint128_t)(b)) % (uint128_t)(m));
    uint64_t RES = res & 0xffffffffffffffff;  
    return RES; 
}


__device__ uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    while (exp) {
        if (exp & 1) res = mulmod(res, base, mod); 
        base = mulmod(base, base, mod); 
        exp >>= 1;
    }
    return res;
}


//any (nxm) X (mxk) 512 bit bigint matmul provided n, m, k are multiples of 4, resulting matrix will have 1024 bits each bigint 
//this kernel can be made faster if we use 64 threads in Z-dim instead of 32 and can reduce the accumulation and load / store
//operation by a factor of 2, but due to shared memory limits on Tesla T4 its not possible
__global__ void _bigint_matmul(uint16_t *A, uint16_t *B, uint16_t *C, int n, int m, int k) {
    int tid_x = threadIdx.x ;//4
    int tid_y = threadIdx.y;//4
    int tid_z = threadIdx.z; //64
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int row_stride = m << 5; 
    uint64_t inv_n = mod - ((mod-1) / 64) ; 
    //51 kB of shared memory used
    __shared__ uint16_t A_shared[4][4][64+1]; //+1 to reduce bank conflicts
    __shared__ uint16_t B_shared[4][4][64+1]; 
    __shared__ uint64_t res_shared[4][4][64+1]; 
    __shared__ uint64_t X1[4][4][64+1];
    __shared__ uint64_t Y1[4][4][64+1]; 
    // use this when you have more shared memory
    // __shared__ uint64_t X2[4][4][64];
    // __shared__ uint64_t Y2[4][4][64]; 
    __shared__ uint64_t Zall[4][4][64+1]; 

    for(int tile_idx = 0; tile_idx < 128; tile_idx++){
        int block_sx1 = (bx * BLOCK_SIZE * (m << 5)); 
        int block_sy1 = tile_idx * (BLOCK_SIZE << 5); 

        int block_sx2 = (tile_idx * BLOCK_SIZE * row_stride); 
        int block_sy2 = by * (BLOCK_SIZE << 5); 
        A_shared[tid_x][tid_y][tid_z] = A[block_sx1 + block_sy1 +  tid_x * (BLOCK_SIZE << 5) + (tid_y << 5) + tid_z];
        B_shared[tid_x][tid_y][tid_z] = B[block_sx2 + block_sy2 +  tid_x * (BLOCK_SIZE << 5) + (tid_y << 5) + tid_z];

        __syncthreads(); 

        for(int muli = 0; muli < 4; muli += 1){
            int tid; 
            uint64_t *X;
            uint64_t *Y; 
            uint64_t *Z;
            X1[tid_x][tid_y][tid_z] = A_shared[tid_x][muli][tid_z]; 
            X1[tid_x][tid_y][tid_z+32] = A_shared[tid_x][muli][tid_z+32];
            Y1[tid_x][tid_y][tid_z] = B_shared[muli][tid_y][tid_z]; 
            Y1[tid_x][tid_y][tid_z+32] = B_shared[muli][tid_y][tid_z+32];


            X = &X1[tid_x][tid_y][0];
            Y = &Y1[tid_x][tid_y][0]; 
            Z = &Zall[tid_x][tid_y][0];
            tid = tid_z;
            uint64_t temp; 
            int idx1, idx2;  
            __syncthreads();

            if(tid < 28){
                idx1 = idx_map[tid][0];
                idx2 = idx_map[tid][1]; 
                temp = X[idx1]; 
                X[idx1] = X[idx2]; 
                X[idx2] = temp;    
                temp = Y[idx1]; 
                Y[idx1] = Y[idx2]; 
                Y[idx2] = temp;    
            }
            __syncthreads(); 


            #pragma unroll
            for (int stage = 1; stage <= 6; stage++) { 
                int length    = 1 << stage;
                int offset    = length >> 1;
                uint64_t wlen = powmod(root, 64 >> stage, mod);
                int k         = tid % offset;
                int pos       = (tid / offset) * length + k;
                uint64_t w    = powmod(wlen, k, mod);
                uint64_t u    = X[pos];
                uint64_t v    = mulmod(X[pos + offset], w, mod); 
                X[pos]        = (u + v) % mod;
                X[pos+offset] = ((u + mod) - v) % mod; 
                u             = Y[pos];
                v             = mulmod(Y[pos + offset], w, mod); 
                Y[pos]        = (u + v) % mod;
                Y[pos+offset] = ((u + mod) - v) % mod; 
                __syncwarp(); 
            }


            Z[tid] = mulmod(X[tid], Y[tid], mod); 
            Z[tid+32] = mulmod(X[tid+32], Y[tid+32], mod); 
            __syncthreads(); 
            if(tid != 0){
                temp = Z[tid]; 
                Z[tid] = Z[64-tid]; 
                Z[64-tid] = temp;    
            }
            __syncthreads(); 
            if(tid < 28){
                idx1 = idx_map[tid][0];
                idx2 = idx_map[tid][1]; 
                temp = Z[idx1]; 
                Z[idx1] = Z[idx2]; 
                Z[idx2] = temp;    
            }
            __syncthreads(); 

            #pragma unroll
            for (int stage = 1; stage <= 6; stage++) {
                int  length   = 1 << stage;
                int  offset   = length >> 1;
                uint64_t wlen = powmod(root, 64 >> stage, mod);
                int k         = tid % offset;
                int pos       = (tid / offset) * length + k;
                uint64_t w    = powmod(wlen, k, mod);
                uint64_t u    = Z[pos];
                uint64_t v    = mulmod(Z[pos + offset], w, mod); 
                Z[pos]        = (u + v) % mod;
                Z[pos+offset] = ((u + mod) - v) % mod;  
                __syncwarp(); 
            } 

            Z[tid] = mulmod(Z[tid], inv_n, mod);
            Z[tid+32] = mulmod(Z[tid+32], inv_n, mod);
            __syncthreads(); 
            res_shared[tid_x][tid_y][tid_z] += Z[tid_z]; 
            res_shared[tid_x][tid_y][tid_z + 32] += Z[tid_z + 32];  
        }
    } 
    //TODO : some sort of parallel scan to accumulate carries + need multiple level of carries for the same
    if(tid_z == 0){
        uint64_t carry = 0; 
        bool flag = true; 
        for(int i = 0; i < 64; i++){
            carry = (res_shared[tid_x][tid_y][i] + carry); 
            res_shared[tid_x][tid_y][i] = ((carry) & 0xffff); 
            carry = carry >> 16; 
            if(res_shared[tid_x][tid_y][i] == 0 and carry == 0){
                flag = false; 
                break; 
            }
        }
        if(flag){
            res_shared[tid_x][tid_y][64] = carry & 0xffff; 
        }
    }
    __syncthreads();
    int block_sx3 = (bx * BLOCK_SIZE * (m << 6)); 
    int block_sy3 = by * (BLOCK_SIZE << 6); 
    __syncthreads(); 
    //first index into the block then index into element of the block 
    C[block_sx3 + block_sy3 +  tid_x * (BLOCK_SIZE << 6) + (tid_y << 6) + tid_z] = (uint16_t)res_shared[tid_x][tid_y][tid_z]; 
    C[block_sx3 + block_sy3 +  tid_x * (BLOCK_SIZE << 6) + (tid_y << 6) + (tid_z + 32)] = (uint16_t)res_shared[tid_x][tid_y][tid_z+32]; 
}


void matmul(uint16_t *A, uint16_t *B, uint16_t *C, int n, int m, int k) {
    dim3 block(4, 4, 32);
    dim3 grid(cdiv(n, BLOCK_SIZE), cdiv(k, BLOCK_SIZE));
    _bigint_matmul<<<grid, block>>>(A, B, C, n, m, k);
    CUDA_CHECK(cudaDeviceSynchronize());
}


//32 length * 32 length -> 64 length (only as of now)
__global__ void _multiply_fft_single_kernel(uint16_t *num1, uint16_t *num2, uint16_t *num3){
    int tid = threadIdx.x; 
    uint64_t inv_root = powmod(root, mod - 2, mod);
    __shared__ uint64_t X[64+1]; //remove bank conflicts
    __shared__ uint64_t Y[64]; 
    __shared__ uint64_t Z[64+1]; 
    uint64_t temp; 
    int idx1, idx2; 
    X[tid] = num1[tid]; 
    Y[tid] = num2[tid]; 
    __syncthreads(); 

    if(tid < 28){
        idx1 = idx_map[tid][0];
        idx2 = idx_map[tid][1]; 
        temp = X[idx1]; 
        X[idx1] = X[idx2]; 
        X[idx2] = temp;    
        temp = Y[idx1]; 
        Y[idx1] = Y[idx2]; 
        Y[idx2] = temp;    
    }
    __syncthreads(); 

    #pragma unroll
    for (int stage = 1; stage <= 6; stage++) {
        int length = 1 << stage;
        int offset   = length >> 1;
        uint64_t wlen = powmod(root, 64 >> stage, mod);
        int k     = tid % offset;
        int pos   = (tid / offset) * length + k;
        uint64_t w = powmod(wlen, k, mod);
        uint64_t u = X[pos];
        uint64_t v = mulmod(X[pos + offset], w, mod); 

        X[pos]         = (u + v) % mod;
        X[pos+offset]    = ((u + mod) - v) % mod; 
        u = Y[pos];
        v = mulmod(Y[pos + offset], w, mod); 
        Y[pos]         = (u + v) % mod;
        Y[pos+offset]    = ((u + mod) - v) % mod; 
        __syncthreads(); 
    }

    Z[tid] = mulmod(X[tid], Y[tid], mod); 
    Z[tid+32] = mulmod(X[tid+32], Y[tid+32], mod); 
    
    __syncthreads(); 

    if(tid != 0){
        temp = Z[tid]; 
        Z[tid] = Z[64-tid]; 
        Z[64-tid] = temp;    
    }
    __syncthreads(); 

    if(tid < 28){
        temp = Z[idx1]; 
        Z[idx1] = Z[idx2]; 
        Z[idx2] = temp;    
    }
    __syncthreads(); 

    #pragma unroll
    for (int stage = 1; stage <= 6; stage++) {
        int  length = 1 << stage;
        int  offset   = length >> 1;
        uint64_t wlen = powmod(root, 64 >> stage, mod);
        int k     = tid % offset;
        int pos   = (tid / offset) * length + k;
        uint64_t w = powmod(wlen, k, mod);
        uint64_t u = Z[pos];
        uint64_t v = mulmod(Z[pos + offset], w, mod); 
        Z[pos]         = (u + v) % mod;
        Z[pos+offset]    = ((u + mod) - v) % mod;  
        __syncthreads(); 
    } 

    uint64_t inv_n = mod - ((mod-1) / 64) ; 
    Z[tid] = mulmod(Z[tid], inv_n, mod);
    Z[tid+32] = mulmod(Z[tid+32], inv_n, mod);

    __syncthreads();    

    //TODO : some sort of parallel scan 
    if(tid == 0){
        uint64_t carry = 0; 
        temp = 0; 
        for(int i = 0; i < 64; i++){
            carry = (Z[i] + carry); 
            num3[i] = (uint16_t)((carry) & 0xffff);
            carry = carry >> 16; 
            if(Z[i] == 0 and carry == 0){
                break; 
            }
        }
    }
}

__global__ void _add_kernel(uint16_t *num1, uint16_t *num2, uint16_t *num3, int n){
    int tid = threadIdx.x;  
    __shared__ uint32_t res[64]; 
    //add vectorized loads and stores
    uint32_t val1 = num1[tid]; 
    uint32_t val2 = num2[tid]; 
    res[tid] = val1 + val2; 
    uint32_t carry = 0; 
    if(tid == 0){
        for(int i = 0; i < n+1; i++){
            res[i] = res[i] + carry; 
            num3[i] = res[i] & 0xffff; 
            carry += (res[i] >> 16); 
        }
        num3[n+1] = carry & 0xffff; 
    }
}


__global__ void _multiply_simple_kernel(uint16_t *num1, uint16_t *num2, uint16_t *num3, int len1, int len2, int outlen, int block_size){
    __shared__ uint16_t out[64+1]; 
    __shared__ uint32_t carry[64]; 
    uint32_t temp_sum; 
    uint32_t lcarry; 
    uint32_t temp; 
    int tid = threadIdx.x; 

    for(int i = 0; i < outlen; i += block_size){
        if (i > outlen) continue; 
        #pragma unroll
        for(int j = 0 ; j < len2; j++){
            if ((tid - j) >= 0 and (tid - j) < len1){   
                temp = num1[tid - j] * num2[j]; 
                temp_sum += temp & 0xffff;
                lcarry += (temp >> 16); 
            }
        }
        out[tid] = temp_sum,
        carry[tid+1] = lcarry; 
        __syncthreads(); 
        carry[tid] += out[tid]; 
        lcarry = 0; 
        temp_sum = 0; 
        tid += block_size; 
        __syncthreads(); 
    } 
    uint16_t val; 
    if(tid == 0){
        for(int i = 0 ; i < 64; i++){
            val = carry[i] >> 16; 
            carry[i] = carry[i] & 0xffff; 
            if (val > 0){
                carry[i+1] += val; 
            }
        }
    }
}



BigInt::BigInt()
{
    this->length = 0; 
    this->limbs = NULL; 
}

BigInt::BigInt(const BigInt& obj)
{
    this->limbs = obj.limbs; 
    this->length = obj.length; 
}

BigInt BigInt::operator*(BigInt const& other)
{
    BigInt val; 
    uint32_t len = next_power_of_2(other.length + this->length); 
    val.length = len;
    cudaMalloc(&val.limbs, len * sizeof(uint16_t)); 
    _multiply_fft_single_kernel<<<1, 32>>>(this->limbs, other.limbs, val.limbs);
    cudaDeviceSynchronize(); 
    return val; 
}


BigInt BigInt::operator+(BigInt const& other)
{
    BigInt val; 
    uint32_t len = max(other.length, this->length);
    if (len % 2 == 1){
        len++; 
    } 
    val.length = len; 
    cudaMalloc(&val.limbs, len * sizeof(uint16_t)); 
    _add_kernel<<<1, len>>>(this->limbs, other.limbs, val.limbs, len); 
    cudaDeviceSynchronize(); 
    return val; 
} 

std::vector<uint16_t> BigInt::get_limbs()
{
    std::vector<uint16_t> cpu_copy(this->length, 0); 
    cudaMemcpy(cpu_copy.data(), this->limbs, sizeof(uint16_t) * this->length, cudaMemcpyDeviceToHost);
    return cpu_copy; 
}

BigInt::BigInt(uint64_t val)
{
    cudaMalloc(&this->limbs, 8); 
    cudaMemcpy(this->limbs, &val, 8, cudaMemcpyHostToDevice); 
    this->length = 4; 
}

BigInt::BigInt(const std::string &inp)
{
    std::vector<uint16_t> temp_limbs = {0}; 
    for (char num : inp){
        uint64_t carry = num - '0'; 
        uint64_t temp = 0; 
        for(int i = 0; i < temp_limbs.size(); i++){
            temp = (uint32_t)(temp_limbs[i]) * 10 + carry; 
            temp_limbs[i] = (uint16_t)(temp & 0xffff); 
            carry = (uint32_t)(temp >> 16); 
        }
        while(carry > 0){
            temp_limbs.push_back(carry & 0xffff); 
            carry >>= 16; 
        }
    }
    if (temp_limbs.size() % 2 == 1){
        temp_limbs.push_back(0); 
    }
    temp_limbs.reserve(next_power_of_2(temp_limbs.size()));
    this->length = temp_limbs.size(); 
    CUDA_CHECK(cudaMalloc(&this->limbs, sizeof(uint16_t) * temp_limbs.size())); 
    CUDA_CHECK(cudaMemcpy(this->limbs, temp_limbs.data(), temp_limbs.size() * sizeof(uint16_t), cudaMemcpyHostToDevice)); 
}

BigInt multiply_simple(BigInt &A, BigInt &B)
{
    BigInt val; 
    val.length = (A.length + B.length - 1); 
    CUDA_CHECK(cudaMalloc(&val.limbs, sizeof(uint16_t) * val.length)); 
    _multiply_simple_kernel<<<1, 4>>>(A.limbs, B.limbs, val.limbs, 
                                      A.length, B.length, val.length, 4);
    return val; 
}
