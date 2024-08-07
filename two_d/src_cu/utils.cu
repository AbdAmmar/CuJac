
#include <stdio.h>


void check_Errors() {
    cudaError_t err = cudaGetLastError();
    printf("CUDA Error ? %s\n", cudaGetErrorString(err));
}

void checkCudaErrors(cudaError_t err, const char* msg, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA Error in %s at line %d\n", file, line);
        printf("%s - %s\n", msg, cudaGetErrorString(err));
        exit(0);
    }
}


extern "C" void checkCudaErrors_C(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s - %s", msg, cudaGetErrorString(err));
        exit(0);
    }
}


__global__ void max_error_kernel(int ntx, int nty, int nty_local, int nWorkers, double h, double *u, double *err) {

    extern __shared__ double cache[];

    int tid;
    int i, cacheIndex;
    int l, ll;
    int j, jj0, jj1;

    double x, y;
    double tmp, err_local;


    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x;
    err_local = 0.0;

    while (tid < nWorkers) {

        jj0 = nty_local * tid;

        for(j = 0; j < nty_local; j++) {

            y = (double) (jj0 + j) * h;

            jj1 = (jj0 + j) * ntx;

            for(l = 0; l < ntx; l++) {

                x = (double) l * h;

                ll = jj1 + l;

                tmp = fabs(u[ll] - x * y * (x - 1.0) * (y - 1.0));
                if(tmp > err_local) {
                    err_local = tmp;
                }
            }
        }

        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = err_local;
    __syncthreads();

    i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            err_local = cache[cacheIndex];
            tmp = cache[cacheIndex + i];
            if(tmp > err_local) {
                cache[cacheIndex] = tmp;
            }
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        err[blockIdx.x] = cache[0];
    }

}


extern "C" void max_error(int nBlocks, int blockSize, size_t size_err, int ntx, int nty, int nty_local, int nWorkers, double h, double *u, double*err) {

    max_error_kernel<<<nBlocks, blockSize, size_err>>>(ntx, nty, nty_local, nWorkers, h, u, err);

}


