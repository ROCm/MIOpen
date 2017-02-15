/* Only works for NCHW
 */

#define W_0 1
#define W_1 c_n * W_0
#define W_2 c_c * W_1
#define W_3 c_h * W_2

#define W_CAT(a, b) a ## b
#define W(n) W_CAT(W_, n)

__kernel void AddTensor(global float *a, 
        global float *c,
        const int c_n, const int c_c, const int c_h, const int c_w,
        const int work_per_wg)
{   
    int gid = get_group_id(0);
    int lid = get_local_id(0);

    float add = a[gid];
    while(lid < W(FIRST_N) * work_per_wg) {
#if FIRST_N == 0
        c[gid*c_c*c_h*c_w + lid] += add;

#elif FIRST_N == 1 // convolution_fwd_bias case
        int o_n = lid / work_per_wg;
        int o_hw = lid % work_per_wg;

        c[o_n*c_c*c_h*c_w + gid*c_h*c_w + o_hw] += add;

#elif FIRST_N == 2
        int o_n = lid / (work_per_wg * c_c);
        int o_c = (lid / work_per_wg) % c_c;
        int o_w = lid % work_per_wg;

        c[o_n*c_c*c_h*c_w + o_c*c_h*c_w + gid*c_w + o_w] += add;

#elif FIRST_N == 3
        int o_n = lid / (c_h * c_c);
        int o_c = (lid / c_h) % c_c;
        int o_h = lid % c_h;
        
        c[o_n*c_c*c_h*c_w + o_c*c_h*c_w + o_h*c_w + gid] += add;

#endif // FIRST_N

        lid += 256;
    }
}


