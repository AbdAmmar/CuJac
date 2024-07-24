
__global__ void communication_kernel(int ntx, int nty_local, int nWorkers, double *u) {


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



extern "C" void communication(int nBlocks, int blockSize, int ntx, int nty_local, int nWorkers, double *u) {

    communication_kernel<<<nBlocks, blockSize>>>(ntx, nty_local, nWorkers, u);


}




