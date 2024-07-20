


__global__ void compute(int ntx, int nty, int nty_local, int n_Workers, double h, double *u, double *u_new) {

    int tid;

    int l, ll;
    int j, jj0, jj1, jj2, jy;

    double x, y, h2;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    h2 = h * h;

    while (tid < n_Workers) {

        jj0 = nty_local * tid;

        jy = jj0 - 2 * tid - 1;

        for(j = 1; j < nty_local-1; j++) {
        
            y = (double) (jy + j) * h;
        
            jj1 = (jj0 + j) * ntx;
        
            for(l = 1; l < ntx-1; l++) {
        
                x = (double) l * h;
        
                ll = jj1 + l;
        
                u_new[ll] = 0.25 * (u[ll - 1] + u[ll + 1] + u[ll - ntx] + u[ll + ntx] - h2 * (2.0 * (x * (x - 1.0) + y * (y - 1.0))));
            }
        }
        
        if(jj0 != 0) {
            jj1 = (jj0 + 1) * ntx;
            jj2 = jj1 - ntx;
            for(l = 0; l < ntx; l++) {
                u_new[jj2 + l] = u_new[jj1 + l];
            }
        } else {
            for(l = 0; l < ntx; l++) {
                u_new[ntx + l] = 0.0;
            }
        }
      
        if(jj0 + nty_local != nty) {
            jj1 = (jj0 + nty_local) * ntx;
            jj2 = jj1 - ntx;
            for(l = 0; l < ntx; l++) {
                u_new[jj1 - l - 1] = u_new[jj2 - l - 1];
            }
        } else {
            jj1 = (nty - 1) * ntx;
            for(l = 0; l < ntx; l++) {
                u_new[jj1 - l - 1] = 0.0;
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

