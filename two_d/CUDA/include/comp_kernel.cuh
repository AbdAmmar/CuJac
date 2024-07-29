#ifndef COMP_KERNEL
#define COMP_KERNEL

__global__ void compute(int ntx, int nty, int nty_local, int nWorkers, double h, double *u_old, double *u_new);

#endif
