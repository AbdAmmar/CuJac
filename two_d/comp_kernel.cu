
__global__ void compute_kernel(int ntx, int nty_local, int n_Workers, double h, double *f, double *u) {

    int tid;
    int l;
    int ii, ll;
    double tmp;

    double *u_tmp;
    size_t size_tmp;

    size_tmp = ntx * nty_local * sizeof(double);

    cudaMalloc(&u_tmp, size_tmp);

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n_Workers) {

        ii = nty_local* tid;

        for(j = 0; j < nty_local; j++) {

            jj0 = ntx * j;
            jj1 = jj0 + ii;

            for(l = 0; l < ntx; l++) {

                ll0 = l + jj0;
                ll1 = l + jj1;

                u_tmp[ll0] = u[ll1];
            }
        }

        for(j = 1; j < nty_local-1; j++) {

            jj0 = ntx * j;
            jj1 = jj0 + ii;

            for(l = 1; l < ntx-1; l++) {

                ll0 = l + jj0;
                ll1 = l + jj1;

                u[ll1] = 0.25 * ( u_tmp[(l - 1) + ntx * j] + u_tmp[(l + 1) + ntx * j]
                                  u_tmp[l + ntx * (j - 1)] + u_tmp[l + ntx * (j + 1)] ) 
                       - h * h * f[ll1];
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

