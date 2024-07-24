#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>



extern "C" void checkCudaErrors(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %f - %f", msg, cudaGetErrorString(err));
        exit(0);
    }
}



__global__ void max_error_kernel(int ntx, int nty, int nty_local, int nWorkers, double h, double *u, double *err) {

    extern __shared__ double cache[];

    int tid;
    int i, cacheIndex;
    int l, ll;
    int j, jj0, jj1, jy;

    double x, y;
    double tmp, err_local;


    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x;
    err_local = 0.0;

    while (tid < nWorkers) {

        jj0 = nty_local * tid;
        jy = jj0 - 2 * tid - 1;

        for(j = 1; j < nty_local-1; j++) {

            y = (double) (jy + j) * h;

            jj1 = (jj0 + j) * ntx;

            for(l = 1; l < ntx-1; l++) {

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



