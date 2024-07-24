#ifndef COMM_KERNEL
#define COMM_KERNEL

__global__ void communication(int ntx, int nty_local, int nWorkers, double *u);

__global__ void naivecopy(int ntx, int nty, int nty_local, int nWorkers, double*u_old, double *u_new);

#endif

