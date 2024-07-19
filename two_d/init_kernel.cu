
__global__ void init(int ntx, int nty_local, int n_Workers, double *u) {


    int tid;
    int j, jj1, l;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n_Workers) {

        for(j = 0; j < nty_local; j++) {

            jj1 = j * ntx;

            for(l = 0; l < ntx; l++) {

                u[l + jj1] = 0.0;

            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

