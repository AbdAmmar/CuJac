#ifndef COMP_KERNEL_2D
#define COMP_KERNEL_2D

__global__ void compute_2d_kernel(int ntx, int nty, int ntx_local, int nty_local, int nWorkers_x, int nWorkers_y, double h, double *u_old, double *u_new);

#endif
