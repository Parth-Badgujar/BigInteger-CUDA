#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>

typedef unsigned __int128 uint128_t; 
#define BLOCK_SIZE 4

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

class BigInt 
{
public: 
    uint16_t* limbs;
    uint32_t length;  
    BigInt(uint64_t val);     
    BigInt(const std::string &inp);
    BigInt(const std::vector<uint32_t> &inp); 
    BigInt();
    BigInt (BigInt& obj); 
    BigInt (const BigInt& obj); 
    BigInt operator*(BigInt const& other); 
    BigInt operator+(BigInt const& other); 
    std::vector<uint16_t> get_limbs(); 
}; 


class CUDAtimer {
public :
    cudaEvent_t start; 
    cudaEvent_t stop; 
    CUDAtimer(){
        cudaEventCreate(&this->start); 
        cudaEventCreate(&this->stop);
    }

    void start_timer(){
        cudaEventRecord(this->start, 0);
    } 

    void stop_timer(){
        cudaEventRecord(this->stop, 0); 
        cudaEventSynchronize(this->stop); 
    }

    float get_time(){
        float ms; 
        cudaEventElapsedTime(&ms, this->start,this->stop); 
        return ms; 
    }

    ~CUDAtimer() {
        cudaEventDestroy(this->start); 
        cudaEventDestroy(this->stop); 
    }
}; 

BigInt multiply_simple(BigInt &A, BigInt &B); 
void matmul(uint16_t *A, uint16_t *B, uint16_t *C, int n, int m, int k); 