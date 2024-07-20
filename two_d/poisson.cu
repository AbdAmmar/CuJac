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
    int blockSize, n_Blocks, streamSize, nWorkers, nStream;

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


    fptr = fopen("param.txt", "r");
    if(fptr != NULL) {

        if(fgets(readString, 100, fptr) != NULL) {
            n = atoi(readString);
        } else {
            printf("Not able to read n\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            nStream = atoi(readString);
        } else {
            printf("Not able to read nStream\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            streamSize = atoi(readString);
        } else {
            printf("Not able to read streamSize\n");
        }

        if(fgets(readString, 100, fptr) != NULL) {
            blockSize = atoi(readString);
        } else {
            printf("Not able to read blockSize\n");
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


    n_Blocks = (streamSize + blockSize - 1) / blockSize;
    nWorkers = min(streamSize * nStream, (n+2)/3);
    printf("nb of streams = %d\n", nStream);
    printf("size of stream = %d\n", streamSize);
    if(nStream*streamSize > n) {
        printf("increase n, or decrease nStream and/or streamSize");
        exit(0);
    }
    printf("size of blocks = %d\n", blockSize);
    printf("nb of blocks = %d\n", n_Blocks);
    printf("nb of workers = %d\n\n", nWorkers);


    ntx = n;
    nty = n + 2 * nWorkers;
    nty_local = n / nWorkers + 2;

    printf("ntx = %d\n", ntx);
    printf("nty = %d\n", nty);
    printf("nty_local = %d\n\n", nty_local);
    if(nty != nty_local*nWorkers) {
        printf("bad set of parameters !");
        exit(0);
    }



    size_u = ntx * nty * sizeof(double);
    printf("Size of d_u = %zu Bytes \n\n", size_u);

    size_err = n_Blocks * sizeof(double);
    printf("Size of d_err = %zu Bytes \n\n", size_err);

    checkCudaErrors(cudaMalloc(&d_u, size_u), "cudaMalloc");
    checkCudaErrors(cudaMalloc(&d_unew, size_u), "cudaMalloc");
    checkCudaErrors(cudaMalloc(&d_err, size_err), "cudaMalloc");

    h_err = (double*) malloc(size_err);
    if(h_err == NULL) {
        fprintf(stderr, "Memory allocation failed for h_err\n");
        exit(0);
    }


    init<<<n_Blocks, blockSize>>>(ntx, nty_local, nWorkers, d_u);
    cudaDeviceSynchronize();

    it = 1;
    while(it <= it_max) {

        //compute<<<n_Blocks, blockSize>>>(ntx, nty, nty_local, nWorkers, h, d_u, d_unew);
        //cudaDeviceSynchronize();
        //naivecopy<<<n_Blocks, blockSize>>>(ntx, nty, nty_local, nWorkers, d_unew, d_u);
        //cudaDeviceSynchronize();
        //communication<<<n_Blocks, blockSize>>>(ntx, nty_local, nWorkers, d_u);
        //cudaDeviceSynchronize();

        if(it%2 != 0) {
            compute<<<n_Blocks, blockSize>>>(ntx, nty, nty_local, nWorkers, h, d_u, d_unew);
            cudaDeviceSynchronize();
            communication<<<n_Blocks, blockSize>>>(ntx, nty_local, nWorkers, d_unew);
            cudaDeviceSynchronize();
        } else {
            compute<<<n_Blocks, blockSize>>>(ntx, nty, nty_local, nWorkers, h, d_unew, d_u);
            cudaDeviceSynchronize();
            communication<<<n_Blocks, blockSize>>>(ntx, nty_local, nWorkers, d_u);
            cudaDeviceSynchronize();
        }

        if(it%it_print == 0) {
            max_error<<<n_Blocks, blockSize, size_err>>>(ntx, nty, nty_local, nWorkers, h, d_u, d_err);
            cudaDeviceSynchronize();
            cudaMemcpy(h_err, d_err, size_err, cudaMemcpyDeviceToHost);
            err = h_err[0];
            for (i = 1; i < n_Blocks; i++) {
                if(err < h_err[i]) {
                    err = h_err[i];
                }
            }
            printf("it = %d/%d, error = %f\n", it, it_max, err);
        }

        it++;
    }

    //int l, j, ii, jj0, jj1;
    //double* h_u;
    //h_u = (double*) malloc(size_u);
    //if(h_u == NULL) {
    //    fprintf(stderr, "Memory allocation failed for h_u\n");
    //    exit(0);
    //}
    //checkCudaErrors(cudaMemcpy(h_u, d_unew, size_u, cudaMemcpyDeviceToHost), "cudaMemcpy");
    //for(l = 0; l < nWorkers; l++){
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
    //free(h_u);


    free(h_err);

    cudaFree(d_u);
    cudaFree(d_unew);
    cudaFree(d_err);

    return 0;
}

