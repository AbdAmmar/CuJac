
__global__ void communication(int ntx, int nty_local, int nWorkers, double *u) {


    int tid;
    int l;
    int jj0, jj1, jj2;
    double tmp;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nWorkers) {

        jj0 = nty_local * tid;

        if(jj0 != 0) {
            jj1 = (jj0 - 1) * ntx;
            jj2 = jj1 + ntx;
            for(l = 0; l < ntx; l++) {
                tmp = u[jj1 + l];
                u[jj1 + l] = u[jj2 + l];
                u[jj2 + l] = tmp;
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}



__global__ void naivecopy(int ntx, int nty, int nty_local, int nWorkers, double*u_old, double *u_new) {

    int tid;
    int l, ll;
    int j, jj0, jj1;

    tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nWorkers) {

        jj0 = nty_local * tid;

        for(j = 0; j < nty_local; j++) {
            jj1 = (jj0 + j) * ntx;
            for(l = 0; l < ntx; l++) {
                ll = jj1 + l;
                u_new[ll] = u_old[ll];
            }
        }

        tid += blockDim.x * gridDim.x;
    }
}





