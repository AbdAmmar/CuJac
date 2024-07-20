#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "utils.cuh"
#include "init_kernel.cuh"
#include "comp_kernel.cuh"
#include "comm_kernel.cuh"




int main() {


    int n;
    int ntx, nty;
    int nty_local;
    int n_Threads, n_Blocks, n_Workers;

    size_t size_u;
    size_t size_err;

    int i, j, ii, jj, jj0, jj1, l;
    int it, it_max, it_print;

    double L, h;

    double* h_u;
    double* d_u;
    double* d_unew;

    double* d_err;
    double* h_err;
    double err;

    FILE *fptr;
    char readString[100];

    int nDevices;
    cudaDeviceProp prop;
    cudaError_t err_cuda;

    err_cuda = cudaGetDeviceCount(&nDevices);
    if(err_cuda != cudaSuccess) printf("%s\n", cudaGetErrorString(err_cuda));
    for (i = 0; i < nDevices; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf("Device: %d/%d\n", i+1, nDevices);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (GHz): %f\n", prop.memoryClockRate/1.0e6);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    //n = 4096;
    //n_Threads = 1;
    //n_Blocks = 1;
    fptr = fopen("param.txt", "r");
    if(fptr != NULL) {

        if(fgets(readString, 100, fptr) != NULL) {
            n = atoi(readString);
        } else {
            printf("Not able to read n\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            n_Threads = atoi(readString);
        } else {
            printf("Not able to read n_Threads\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            n_Blocks = atoi(readString);
        } else {
            printf("Not able to read n_Blocks\n");
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


    n_Workers = n_Threads * n_Blocks;
    printf("nb on threads = %d\n", n_Threads);
    printf("nb on blocks = %d\n", n_Blocks);
    printf("nb on workers = %d\n", n_Workers);


    ntx = n;
    nty = n + 2*n_Workers;
    nty_local = n / n_Workers + 2;
    if((nty_local*n_Workers - nty) != 0) {
        printf("Unconsistent dimensions\n");
        exit(0);
    }

    printf("ntx = %d\n", ntx);
    printf("nty = %d\n", nty);
    printf("nty_local = %d\n\n", nty_local);

    size_u = ntx * nty * sizeof(double);
    printf("Size of d_u = %zu Bytes \n\n", size_u);

    size_err = n_Blocks * sizeof(double);
    printf("Size of d_err = %zu Bytes \n\n", size_err);

    checkCudaErrors(cudaMalloc(&d_u, size_u), "cudaMalloc");
    checkCudaErrors(cudaMalloc(&d_unew, size_u), "cudaMalloc");
    checkCudaErrors(cudaMalloc(&d_err, size_err), "cudaMalloc");

    h_u = (double*) malloc(size_u);
    if(h_u == NULL) {
        fprintf(stderr, "Memory allocation failed for h_u\n");
        exit(0);
    }

    h_err = (double*) malloc(size_err);
    if(h_err == NULL) {
        fprintf(stderr, "Memory allocation failed for h_err\n");
        exit(0);
    }


    init<<<n_Blocks, n_Threads>>>(ntx, nty_local, n_Workers, d_u);
    checkCudaErrors(cudaGetLastError(), "Kernel init launch failed");
    cudaDeviceSynchronize();

    it = 0;
    while(it < it_max) {

        //printf("it = %d/%d\n", it, it_max);

        compute<<<n_Blocks, n_Threads>>>(ntx, nty, nty_local, n_Workers, h, d_u, d_unew);
        checkCudaErrors(cudaGetLastError(), "Kernel compute launch failed");
        cudaDeviceSynchronize();

        //checkCudaErrors(cudaMemcpy(h_u, d_unew, size_u, cudaMemcpyDeviceToHost), "cudaMemcpy");
        //for(l = 0; l < n_Workers; l++){
        //    jj0 = l * nty_local;
        //    for (j = 1; j < nty_local-1; j++) {
        //        jj1 = (jj0 + j) * ntx;
        //        for (i = 0; i < ntx; i++) {
        //            ii = jj1 + i;
        //            printf("%f  ", h_u[ii]);
        //        }
        //        printf("\n");
        //    }
        //    printf("\n");
        //}
        //printf("\n");

        naivecopy<<<n_Blocks, n_Threads>>>(ntx, nty, nty_local, n_Workers, d_unew, d_u);
        cudaDeviceSynchronize();

        communication<<<n_Blocks, n_Threads>>>(ntx, nty_local, n_Workers, d_u);
        cudaDeviceSynchronize();

        max_error<<<n_Blocks, n_Threads, size_err>>>(ntx, nty, nty_local, n_Workers, h, d_u, d_err);
        cudaDeviceSynchronize();

        cudaMemcpy(h_err, d_err, size_err, cudaMemcpyDeviceToHost);
        err = h_err[0];
        for (i = 1; i < n_Blocks; i++) {
            if(err < h_err[i]) {
                err = h_err[i];
            }
        }

        if(it%it_print == 0) {
            printf("it = %d/%d, error = %f\n", it, it_max, err);
        }

        it++;
    }


    checkCudaErrors(cudaMemcpy(h_u, d_unew, size_u, cudaMemcpyDeviceToHost), "cudaMemcpy");

    
    cudaFree(d_u);
    cudaFree(d_unew);
    cudaFree(d_err);
    
    for(l = 0; l < n_Workers; l++){
        jj0 = l * nty_local;
        for (j = 1; j < nty_local-1; j++) {
            jj1 = (jj0 + j) * ntx;
            for (i = 0; i < ntx; i++) {
                ii = jj1 + i;
                printf("%f  ", h_u[ii]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");

    //printf("%f  \n", h_u[0]);
    free(h_u);
    free(h_err);

    return 0;
}

