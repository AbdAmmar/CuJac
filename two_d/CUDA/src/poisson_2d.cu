#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "utils.cuh"
#include "init_kernel_2d.cuh"
#include "comp_kernel_2d.cuh"




int main() {


    int n;
    int ntx, ntx_local;
    int nty, nty_local;
    int blockxSize, nxBlocks, nWorkers_x;
    int blockySize, nyBlocks, nWorkers_y;

    size_t size_u;
    size_t size_err;

    int i;
    int it, it_max, it_print;

    double L, h;

    double* d_u;
    double* d_unew;

    double* d_err;
    double* h_err;
    double err;

    FILE *fptr;
    char readString[100];

    int nDevices;
    cudaDeviceProp prop;

    cudaEvent_t start, stop;
    float tt;

    checkCudaErrors(cudaEventCreate(&start), "cudaEventCreate", __FILE__, __LINE__);
    checkCudaErrors(cudaEventCreate(&stop), "cudaEventCreate",  __FILE__, __LINE__);

    checkCudaErrors(cudaEventRecord(start), "cudaEventRecord", __FILE__, __LINE__);

    nDevices = 0;
    checkCudaErrors(cudaGetDeviceCount(&nDevices), "cudaGetDeviceCount)", __FILE__, __LINE__);
    if(nDevices == 0) {
        printf("no available GPU(s)\n");
        exit(0);
    } else {
        printf("Detected %d GPU(s)\n", nDevices);
    }
    for (i = 0; i < nDevices; i++) {
        checkCudaErrors(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties", __FILE__, __LINE__);
        printf("\nDevice %d/%d: \"%s\"\n", i+1, nDevices, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Memory Clock Rate (GHz): %f\n", prop.memoryClockRate/1.0e6);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Global Memory (GB): %f\n", prop.totalGlobalMem/8.0e9);
        printf("  Constant Memory (Bytes): %zu\n", prop.totalConstMem);
        printf("  Max mem pitch: %ld\n", prop.memPitch);
        printf("  Texture Alignment: %ld\n", prop.textureAlignment);
        printf("  Warp Size : %d\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("  Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("  Registers per mp: %d\n\n", prop.regsPerBlock);
    }


    fptr = fopen("param_2d.txt", "r");
    if(fptr != NULL) {

        if(fgets(readString, 100, fptr) != NULL) {
            n = atoi(readString);
        } else {
            printf("Not able to read n\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            nxBlocks = atoi(readString);
        } else {
            printf("Not able to read nxBlocks\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            nyBlocks = atoi(readString);
        } else {
            printf("Not able to read nyBlocks\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            blockxSize = atoi(readString);
        } else {
            printf("Not able to read blockxSize\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            blockySize = atoi(readString);
        } else {
            printf("Not able to read blockySize\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            it_max = atoi(readString);
        } else {
            printf("Not able to read it_max\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            it_print = atoi(readString);
        } else {
            printf("Not able to read it_print\n");
        }

    } else {
        printf("Not able to open the file param.txt\n");
    }
    fclose(fptr);

    L = 1.0;
    h = L / (double) (n-1);

    printf("nb on grid points = %d x %d\n", n, n);
    printf("dim of grid = %.1f x %.1f\n", L, L);
    printf("step = %f\n\n", h);


    nWorkers_x = nxBlocks * blockxSize;
    printf("nb of blocks (x) = %d\n", nxBlocks);
    printf("size of blocks (x) = %d\n", blockxSize);
    printf("nb of workers (x) = %d\n\n", nWorkers_x);
    if(nWorkers_x > n) {
        printf("increase n, or decrease nxBlocks and/or blockxSize");
        exit(0);
    }

    nWorkers_y = nyBlocks * blockySize;
    printf("nb of blocks (y) = %d\n", nyBlocks);
    printf("size of blocks (y) = %d\n", blockySize);
    printf("nb of workers (y) = %d\n\n", nWorkers_y);
    if(nWorkers_y > n) {
        printf("increase n, or decrease nyBlocks and/or blockySize");
        exit(0);
    }


    ntx = n;
    nty = n;
    ntx_local = n / nWorkers_x;
    nty_local = n / nWorkers_y;

    printf("ntx = %d\n", ntx);
    printf("nty = %d\n", nty);
    printf("ntx_local = %d\n", ntx_local);
    printf("nty_local = %d\n\n", nty_local);
    if(ntx != ntx_local*nWorkers_x) {
        printf("bad set of parameters (x) !");
        exit(0);
    }
    if(nty != nty_local*nWorkers_y) {
        printf("bad set of parameters (y) !");
        exit(0);
    }


    size_u = ntx * nty * sizeof(double);
    printf("Size of d_u = %.2f MB \n\n", (double)size_u/(1024.0*1024.0));

    size_err = nyBlocks * sizeof(double);

    checkCudaErrors(cudaMalloc(&d_u, size_u), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc(&d_unew, size_u), "cudaMalloc", __FILE__, __LINE__);

    checkCudaErrors(cudaMalloc(&d_err, size_err), "cudaMalloc", __FILE__, __LINE__);



    h_err = (double*) malloc(size_err);
    if(h_err == NULL) {
        fprintf(stderr, "Memory allocation failed for h_err\n");
        exit(0);
    }


    dim3 dimGrid(nxBlocks, nyBlocks, 1);
    dim3 dimBlock(blockxSize, blockySize, 1);


    init_2d_kernel<<<dimGrid, dimBlock>>>(ntx, ntx_local, nty_local, nWorkers_x, nWorkers_y, d_u);
    checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
    checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);


    it = 1;
    while(it <= it_max) {

        if(it%2 != 0) {
            compute_2d_kernel<<<dimGrid, dimBlock>>>(ntx, nty, ntx_local, nty_local, nWorkers_x, nWorkers_y, h, d_u, d_unew);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        } else {
            compute_2d_kernel<<<dimGrid, dimBlock>>>(ntx, nty, ntx_local, nty_local, nWorkers_x, nWorkers_y, h, d_unew, d_u);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
        }

        if(it%it_print == 0) {
            max_error_kernel<<<nyBlocks, blockySize, size_err>>>(ntx, nty, nty_local, nWorkers_y, h, d_u, d_err);
            checkCudaErrors(cudaDeviceSynchronize(), "cudaDeviceSynchronize", __FILE__, __LINE__);
            checkCudaErrors(cudaGetLastError(), "cudaGetLastError", __FILE__, __LINE__);
            checkCudaErrors(cudaMemcpy(h_err, d_err, size_err, cudaMemcpyDeviceToHost), "cudaMemcpy", __FILE__, __LINE__);
            err = h_err[0];
            for (i = 1; i < nyBlocks; i++) {
                if(err < h_err[i]) {
                    err = h_err[i];
                }
            }
            printf("it = %d/%d, error = %f\n", it, it_max, err);
        }

        it++;
    }

    free(h_err);


    checkCudaErrors(cudaFree(d_u), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_unew), "cudaFree", __FILE__, __LINE__);
    checkCudaErrors(cudaFree(d_err), "cudaFree", __FILE__, __LINE__);

    checkCudaErrors(cudaEventRecord(stop), "cudaEventRecord", __FILE__, __LINE__);
    checkCudaErrors(cudaEventSynchronize(stop), "cudaEventSynchronize", __FILE__, __LINE__);
    checkCudaErrors(cudaEventElapsedTime(&tt, start, stop), "cudaEventElapsedTime", __FILE__, __LINE__);

    printf("Ellapsed time = %.3f sec", tt / 1000.0f);

    return 0;
}

