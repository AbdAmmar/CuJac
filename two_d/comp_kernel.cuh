#ifndef COMP_KERNEL
#define COMP_KERNEL

__global__ void compute(int ntx, int nty, int nty_local, int n_Workers, double h, double *u, double *u_new);

#endif
