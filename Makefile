gmp_bench:
	g++ -I./gmp-6.3.0 -fopenmp -L./gmp-6.3.0/.libs -o ./bin/gmp_bench -c ./src/gmp_bench.cpp -lgmp

gpu_bench:
	nvcc -Xcompiler -fPIC -I./include -shared -o ./lib/libbigint.so ./src/bigint.cu 
	nvcc -I./include -L./lib -lbigint -o ./bin/gpu_bench ./src/gpu_bench.cu 

