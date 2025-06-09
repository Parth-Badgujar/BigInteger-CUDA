#include <iostream>
#include <gmpxx.h>
#include <vector>
#include <chrono>
#include <omp.h>

gmp_randclass rng(gmp_randinit_default);

typedef std::vector<std::vector<mpz_class>> Matrix; 

void benchmark_addition(int bits, int iters){
    mpz_class num1 = rng.get_z_bits(bits);
    mpz_class num2 = rng.get_z_bits(bits); 
    mpz_class num3; 

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iters; i++){
        num3 = num1 + num2; 
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
    std::cout << "[*] " << bits << " bits ADD " << iters << " iters : " << time_taken.count() << "ms\n"; 
}

void benchmark_multiplication(int bits, int iters){
    mpz_class num1 = rng.get_z_bits(bits);
    mpz_class num2 = rng.get_z_bits(bits); 
    mpz_class num3; 

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iters; i++){
        num3 = num1 * num2; 
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
    std::cout << "[*] " << bits << " bits MUL " << iters << " iters : " << time_taken.count() << "ms\n"; 
}

void matmul(Matrix &Mat1, Matrix &Mat2, Matrix &Mat3, int n, int m, int k){
    int i, j, l; 
    #pragma omp parallel for private(i, l, j) shared(Mat1, Mat2, Mat3)
    for(i = 0; i < n; i++){
        for(l = 0; l < k; l++){ 
            for(j = 0; j < m; j++){ 
                Mat3[i][l] += Mat1[i][j] * Mat2[j][l]; 
            }
        }
    }
}

void benchmark_matmul(int bits, int n, int m, int k, int iters){ //(nxm) X (mxk) -> (nxk)
    Matrix Mat1(n); 
    Matrix Mat2(m);
    Matrix Mat3(m);
    std::string num1 = "44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444"; 
    std::string num2 = "33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333"; 

    for(int i = 0 ; i < n; i++){
        for(int j = 0; j < m; j++){
            mpz_class NUM1(num1); 
            Mat1[i].push_back(NUM1); 
        }
    }  
    for(int i = 0 ; i < m; i++){
        for(int j = 0; j < k; j++){
            mpz_class NUM2(num2); 
            Mat2[i].push_back(NUM2);  
        }
    }  
    for(int i = 0 ; i < m; i++){
        for(int j = 0; j < k; j++){
            mpz_class NUM3(0); 
            Mat3[i].push_back(NUM3); 
        }
    } 

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iters; i++){
        matmul(Mat1, Mat2, Mat3, 
                n, m, k); 
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 

    std::cout << Mat3[0][0].get_str(10) << "\n"; 

    std::cout << "[*] " << bits << " bits MATMUL("
              << n << "x" << m << "x" << k << ") "
              << iters << " iters : " << time_taken.count() << "ms\n"; 
}

int main() {
    
    int bit_lengths[5] = {
            128, 
            256, 
            1024, 
            2048, 
            4096
    }; 

    for(int i = 0 ; i < 5; i++){
        benchmark_addition(bit_lengths[i], 1000000);
    }
    for(int i = 0 ; i < 5; i++){
        benchmark_multiplication(bit_lengths[i], 100000);
    }

    // for(int i = 0 ; i < 5; i++){
    benchmark_matmul(512, 16, 16, 16, 10000);
    // }

    return 0;
}
