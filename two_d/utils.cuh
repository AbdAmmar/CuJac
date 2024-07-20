#ifndef UTILS
#define UTILS

void checkCudaErrors(cudaError_t err, const char* msg);

__global__ void max_error(int ntx, int nty, int nty_local, int nWorkers, double h, double *u, double *err);

#endif
