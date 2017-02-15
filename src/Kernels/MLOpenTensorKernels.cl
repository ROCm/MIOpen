/* Only works for NCHW
 */

__kernel void AddTensor(global float *a, 
        const int a_c, const int a_h, const int a_w,
        const int a_nstride, a_cstride,
        global float *c,
        const int c_c, const int c_h, const int c_w,
        const int c_nstride, c_cstride,
        const unsigned int bitmap, const int work_per_wg)
{   
    int gid = get_group_id(0);
    int lid = get_local_id(0);

    float add = a[gid];

    int o_h_div = bitmap & (1 << 0) ? 1 : c_w;
    int o_c_div = o_h_div * (bitmap & (1 << 1) ? 1 : c_h);
    int o_n_div = o_c_div * (bitmap & (1 << 2) ? 1 : c_c);

    while(lid < work_per_wg) {
        int o_w = (bitmap & (1 << 0)) ? gid % a_w               : lid % c_w;
        int o_h = (bitmap & (1 << 1)) ? (gid / a_w) % a_h       : (lid / o_h_div) % c_h;
        int o_c = (bitmap & (1 << 2)) ? (gid / a_cstride) % a_c : (lid / o_c_div) % c_c;
        int o_n = (bitmap & (1 << 3)) ? gid / a_nstride         : lid / o_n_div;

        c[o_n*c_nstride + o_c*c_cstride + o_h*c_w + o_w] += add;

        lid += get_local_size(0);
    }
}


