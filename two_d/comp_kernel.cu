


__global__ void compute_kernel(int ntx, int nty, int nty_local, int n_Workers, double h, double *u) {

    int tid;

    int l, ll;
    int j, jj0, jj1, jj2, jy;

    double *u_tmp;
    size_t size_tmp;

    double x, y, h2;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    h2 = h * h;

    size_tmp = ntx * nty_local * sizeof(double);

    while (tid < n_Workers) {

        // TODO
        // single big allocate ?
        cudaMalloc(&u_tmp, size_tmp);

        jj0 = nty_local * tid;

        for(j = 0; j < nty_local; j++) {

            jj1 = j * ntx;
            jj2 = (jj0 + j) * ntx;

            for(l = 0; l < ntx; l++) {

                u_tmp[jj1 + l] = u[jj2 + l];

            }
        }

        jy = jj0 - 2 * tid - 1;

        for(j = 1; j < nty_local-1; j++) {
        
            y = (double) (jy + j) * h;
        
            jj1 = j * ntx;
            jj2 = (jj0 + j) * ntx;
        
            for(l = 1; l < ntx-1; l++) {
        
                x = (double) l * h;
        
                ll = jj1 + l;
        
                u[jj2 + l] = 0.25 * ( u_tmp[ll - 1] + u_tmp[ll + 1] + u_tmp[ll - ntx] + u_tmp[ll + ntx] 
                                    - h2 * (2.0 * (x * (x - 1) + y * (y - 1))) );
            }
        }
        
        free(u_tmp);

        if(jj0 != 0) {
            jj1 = (jj0 + 1) * ntx;
            jj2 = jj1 - ntx;
            for(l = 0; l < ntx; l++) {
                u[jj2 + l] = u[jj1 + l];
            }
        } else {
            for(l = 0; l < ntx; l++) {
                u[ntx + l] = 0.0;
            }
        }
      
        if(jj0 + nty_local != nty) {
            jj1 = (jj0 + nty_local) * ntx;
            jj2 = jj1 - ntx;
            for(l = 0; l < ntx; l++) {
                u[jj1 - l - 1] = u[jj2 - l - 1];
            }
        } else {
            jj1 = (nty - 1) * ntx;
            for(l = 0; l < ntx; l++) {
                u[jj1 - l - 1] = 0.0;
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}

