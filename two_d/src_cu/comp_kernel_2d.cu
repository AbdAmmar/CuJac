


__global__ void compute_2d_kernel(int ntx, int nty, int ntx_local, int nty_local, int nWorkers_x, int nWorkers_y, double h, double *u_old, double *u_new) {

    int tid_x, tid_y;

    int js, je, jj_check;
    int is, ie, ii_check;

    int j, jj0, jj1;
    int i, ii0;
    int ll;

    //int do_bc_top;
    //int do_bc_bottom;
    //int do_bc_left;
    //int do_bc_right;

    double x, y, y_tmp;

    const double h_ct = 2.0 * h * h;

    tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    //do_bc_top = 0;
    //do_bc_bottom = 0;
    //do_bc_left = 0;
    //do_bc_right = 0;

    if(tid_y != 0) {
        js = 0;
    } else {
        js = 1;
        //do_bc_top = 1;
    }

    if(tid_x != 0) {
        is = 0;
    } else {
        is = 1;
        //do_bc_left = 1;
    }

    jj_check = nty - nty_local;
    //j_f = ntx * nty;

    ii_check = ntx - ntx_local;
    //i_f = ntx * nty;

    while (tid_y < nWorkers_y) {

        jj0 = nty_local * tid_y;

        if(jj0 != jj_check) {
            je = nty_local;
        } else {
            je = nty_local - 1;
            //do_bc_bottom = 1;
        }

        while (tid_x < nWorkers_x) {

            ii0 = ntx_local * tid_x;

            if(ii0 != ii_check) {
                ie = ntx_local;
            } else {
                ie = ntx_local - 1;
                //do_bc_right = 1;
            }

    
            for(j = jj0 + js; j < jj0 + je; j++) {
            
                y = (double) j * h;
                y_tmp = y * (y - 1.0);
            
                jj1 = j * ntx;
            
                for(i = ii0 + is; i < ii0 + ie; i++) {
            
                    x = (double) i * h;
            
                    ll = jj1 + i;
            
                    u_new[ll] = 0.25 * ( u_old[ll - 1] + u_old[ll + 1] + u_old[ll - ntx] + u_old[ll + ntx] 
                                       - h_ct * (x * (x - 1.0) + y_tmp) ) ;
                }
            }
            
            tid_x += blockDim.x * gridDim.x;
        }

        tid_y += blockDim.y * gridDim.y;
    }

    // TODO
    //if(do_bc_top == 1) {
    //    for(l = 0; l < ntx_local; l++) {
    //        u_new[l] = 0.0;
    //    }
    //}

    //if(do_bc_bottom = 1) {
    //    for(l = 0; l < ntx; l++) {
    //        u_new[j_f - l - 1] = 0.0;
    //    }
    //}

}



extern "C" void compute_2d(dim3 grid_dim, dim3 block_dim, 
                           int ntx, int nty, int ntx_local, int nty_local, int nWorkers_x, int nWorkers_y, double h, double *u_old, double *u_new) {

    compute_2d_kernel<<<grid_dim, block_dim>>>(ntx, nty, ntx_local, nty_local, nWorkers_x, nWorkers_y, h, u_old, u_new);

}

