# BigInteger-CUDA


### Download and Build GMP 6.3.0
```bash
chmod +x ./setup.sh
./setup.sh
``` 

### Set LD_LIBRARY_PATH 
```bash
export LD_LIBRARY_PATH=./lib:./gmp-6.3.0/.libs:$LD_LIBRARY_PATH
```

### Build BigInt CUDA
```bash 
make gpu_becch

chmod +x ./bin/gpu_bench
./bin/gpu_bench
```

### Build GMP Code 
```bash 
make gmp_bench

chmod +x ./bin/gmp_bench
./bin/gmp_bench
```