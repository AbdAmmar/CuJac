


__global__ void compute_kernel(int ntx, int nty, int nty_local, int nWorkers, double h, double *u_old, double *u_new) {

    int tid;

    int l, ll;
    int js, je, jj_check, jf;
    int j, jj0, jj1;

    double x, y, y_tmp;

    const double h_ct = 2.0 * h * h;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid != 0) {
        js = 0;
    } else {
        js = 1;
        for(l = 0; l < ntx; l++) {
            u_new[l] = 0.0;
        }
    }

    jj_check = nty - nty_local;
    jf = ntx * nty;

    while (tid < nWorkers) {

        jj0 = nty_local * tid;

        if(jj0 != jj_check) {
            je = nty_local;
        } else {
            je = nty_local - 1;
            for(l = 0; l < ntx; l++) {
                u_new[jf - l - 1] = 0.0;
            }
        }

        for(j = js; j < je; j++) {
        
            y = (double) (jj0 + j) * h;
            y_tmp = y * (y - 1.0);
        
            jj1 = (jj0 + j) * ntx;
        
            for(l = 1; l < ntx-1; l++) {
        
                x = (double) l * h;
        
                ll = jj1 + l;
        
                u_new[ll] = 0.25 * ( u_old[ll - 1] + u_old[ll + 1] + u_old[ll - ntx] + u_old[ll + ntx] 
                                   - h_ct * (x * (x - 1.0) + y_tmp) ) ;
            }
        }
        
        tid += blockDim.x * gridDim.x;
    }
}

extern "C" void compute(int nBlocks, int blockSize, int ntx, int nty, int nty_local, int nWorkers, double h, double *u_old, double *u_new) {

    compute_kernel<<<nBlocks, blockSize>>>(ntx, nty, nty_local, nWorkers, h, u_old, u_new);

}

