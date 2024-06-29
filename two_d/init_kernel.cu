
__global__ void initalize_kernel(int ntx, int nty_local, int n_Workers, double h, double *f, double *u) {


    int tid;
    int j, k, l;
    int ii0, ii1, ii2;
    int ll0, ll1, ll2, ll3;
    double x, y;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < n_Workers) {

        ii0 = nty_local * tid;
        ii1 = ii0 + nty_local + 1;
        ii2 = 2 * tid;

        for(j = 1; j <= nty_local; j++) {

            k = ii0 + j;
            y = __int2double_rn(k) * h;

            ll0 = (k + ii2) * ntx;
            for(l = 0; l < ntx; l++) {

                ll1 = l + ll0;

                x = __int2double_rn(l+1) * h;

                // TODO : call a general function for the source
                f[ll1] = 2.0 * (x * x - x + y * y - y);
                u[ll1] = 0.0;
            }

            ll0 = ii0 + ii2;
            ll1 = ii1 + ii2;
            for(l = 0; l < ntx; l++) {

                ll2 = l + ll0;
                ll3 = l + ll1;

                f[ll2] = f[ll2+1];
                f[ll3] = f[ll3-1];
               
                u[ll2] = u[ll2+1];
                u[ll3] = u[ll3-1];
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

