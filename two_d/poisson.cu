

int main(void) {

    int nx, ny;
    int ntx, nty;
    int n_Workers, nty_local;

    int it, it_max;

    int size_u;
    int n_Blocks, n_Threads;

    double Lx, h;

    double *d_u;
    double *d_f;


    nx = 1024;
    ny = 1024;

    Lx = 1.0;

    h = Lx / double(nx);

    n_Threads = 32;
    n_Blocks = 1;

    n_Workers = n_Threads * n_Blocks;

    nty_local = ny / n_Workers;

    if((nty_local*n_Workers - ny) != 0)
        break;

    ntx = nx + 2;
    nty = ny + 2*n_Workers;

    it_max = 10;

    size_u = ntx * nty * sizeof(double);


    cudaMalloc(&d_u, size_u); // solution
    cudaMalloc(&d_f, size_u); // source

    // n_Blocks = imin((n_Workers+n_Threads-1) / n_Threads, n_Blocks_max);


    initalize_kernel<<<n_Blocks, n_Threads>>>(ntx, nty_local, n_Workers, h, d_f, d_u);

    it = 0;
    while(it < it_max) {
        it++;
        communication_kernel<<<n_Blocks, n_Threads>>>(ntx, nty_local, n_Workers, d_u);
        compute_kernel<<<n_Blocks, n_Threads>>>(ntx, nty_local, n_Workers, h, d_f, d_u);
    }


    // TODO compare with exact solution
    cudaFree(d_u);

    return 0;
}

