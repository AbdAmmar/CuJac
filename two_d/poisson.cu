#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "utils.cuh"
#include "init_kernel.cuh"

//__global__ void init(int ntx, int nty_local, int n_Workers, double *u) {
//
//
//    int tid;
//    int j, jj0, jj1, l;
//
//    tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//    //while (tid < n_Workers) {
//    while (tid < 1) {
//
//        jj0 = nty_local * tid;
//
//        for(j = 0; j < nty_local; j++) {
//
//            jj1 = (jj0 + j) * ntx;
//
//            for(l = 0; l < ntx; l++) {
//
//                u[l + jj1] = 1.0;
//
//            }
//        }
//
//        tid += blockDim.x * gridDim.x;
//    }
//}




int main() {

    int n;
    int ntx, nty;
    int nty_local;
    int n_Threads, n_Blocks, n_Workers;

    size_t size_u;

    int i, j, ii, jj, jj0, jj1, l;
    int it, it_max, it_print;

    double L, h;

    double* h_u;
    double* d_u;

    double* d_err_i;
    double err;

    FILE *fptr;
    char readString[100];

    n = 4096;
    n_Threads = 1;
    n_Blocks = 1;
    //fptr = fopen("param.txt", "r");
    //if(fptr != NULL) {

    //    if(fgets(readString, 100, fptr) != NULL) {
    //        n = atoi(readString);
    //    } else {
    //        printf("Not able to read n\n");
    //    }

    //    if(fgets(readString, 100, fptr) != NULL) {
    //        n_Threads = atoi(readString);
    //    } else {
    //        printf("Not able to read n_Threads\n");
    //    }

    //    if(fgets(readString, 100, fptr) != NULL) {
    //        n_Blocks = atoi(readString);
    //    } else {
    //        printf("Not able to read n_Blocks\n");
    //    }

    //    if(fgets(readString, 100, fptr) != NULL) {
    //        it_max = atoi(readString);
    //    } else {
    //        printf("Not able to read it_max\n");
    //    }

    //    if(fgets(readString, 100, fptr) != NULL) {
    //        it_print = atoi(readString);
    //    } else {
    //        printf("Not able to read it_print\n");
    //    }

    //} else {
    //    printf("Not able to open the file param.txt\n");
    //}
    //fclose(fptr);


    L = 1.0;
    h = L / (double) (n-1);

    //printf("nb on grid points = %d x %d\n", n, n);
    //printf("dim of grid = %.1f x %.1f\n", L, L);
    //printf("step = %f\n\n", h);


    n_Workers = n_Threads * n_Blocks;
    //printf("nb on threads = %d\n", n_Threads);
    //printf("nb on blocks = %d\n", n_Blocks);
    //printf("nb on workers = %d\n", n_Workers);


    ntx = n;
    nty = n + 2*n_Workers;
    nty_local = n / n_Workers + 2;
    //if((nty_local*n_Workers - nty) != 0) {
    //    printf("Unconsistent dimensions\n");
    //    exit(0);
    //}

    //printf("ntx = %d\n", ntx);
    //printf("nty = %d\n", nty);
    //printf("nty_local = %d\n\n", nty_local);

    size_u = ntx * nty * sizeof(double);
    //printf("Size of d_u = %zu Bytes \n\n", size_u);


    checkCudaErrors(cudaMalloc(&d_u, size_u), "cudaMalloc");
    //cudaMalloc(&d_err_i, n_Workers * sizeof(double));

    h_u = (double*) malloc(size_u);
    if(h_u == NULL) {
        fprintf(stderr, "Memory allocation failed for h_u\n");
        exit(0);
    }


    init<<<n_Blocks, n_Threads>>>(ntx, nty_local, n_Workers, d_u);
    checkCudaErrors(cudaGetLastError(), "Kernel launch failed");

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError(), "Kernel launch failed");

    //it = 0;
    //while(it < it_max) {
    //    compute_kernel<<<n_Blocks, n_Threads>>>(ntx, nty, nty_local, h, d_u);
    //    it++;
    //}


  
    checkCudaErrors(cudaMemcpy(h_u, d_u, size_u, cudaMemcpyDeviceToHost), "cudaMemcpy");

    
    cudaFree(d_u);
    //cudaFree(d_err_i);
    
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

    printf("%f  \n", h_u[0]);
    free(h_u);


    return 0;
}

