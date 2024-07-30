#ifndef INIT_KERNEL_2D
#define INIT_KERNEL_2D

__global__ void init_2d_kernel(int ntx, int ntx_local, int nty_local, int nWorkers_x, int nWorkers_y, double *u);

#endif

