#ifndef UTILS
#define UTILS

void checkCudaErrors(cudaError_t err, const char* msg, const char* file, int line);

void check_Errors();
__global__ void max_error_kernel(int ntx, int nty, int nty_local, int nWorkers, double h, double *u, double *err);

#endif
