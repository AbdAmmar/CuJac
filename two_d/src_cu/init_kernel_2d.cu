
__global__ void init_2d_kernel(int ntx, int ntx_local, int nty_local, int nWorkers_x, int nWorkers_y, double *u) {


    int tid_x;
    int tid_y;

    int j, jj0, jj1;
    int i, ii0;
  
    int js, je;
    int is, ie;

    tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    while (tid_y < nWorkers_y) {

        jj0 = tid_y * nty_local;
        js = jj0;
        je = jj0 + nty_local;

        while (tid_x < nWorkers_x) {

            ii0 = tid_x * ntx_local;
            is = ii0;
            ie = ii0 + ntx_local;

            for(j = js; j < je; j++) {

                jj1 = j * ntx;

                for(i = is; i < ie; i++) {

                    u[i + jj1] = 0.0;

                }
            }

            tid_x += blockDim.x * gridDim.x;

        }

        tid_y += blockDim.y * gridDim.y;
    }
}


extern "C" void init_2d(dim3 grid_dim, dim3 block_dim,
                        int ntx, int ntx_local, int nty_local, int nWorkers_x, int nWorkers_y, double *u) {

    init_2d_kernel<<<grid_dim, block_dim>>>(ntx, ntx_local, nty_local, nWorkers_x, nWorkers_y, u);

}

