
__global__ void communication_kernel(int ntx, int nty_local, int n_Workers, double *u) {


    int tid;
    int l;
    int ii, ll;
    double tmp;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n_Workers) {

        ii = nty_local * tid;

        for(l = 0; l < ntx; l++) {

            ll = ii + l

            tmp = u[ll];
            u[ll] = u[ll-1];
            u[ll-1] = tmp;
        }

        tid += blockDim.x * gridDim.x;
    }
}

