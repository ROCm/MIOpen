__kernel void cl_gemm_generate_amap(__global uint* indices, uint m, uint x, uint bs, uint ntidx)
{
    uint dimx = m * bs;
    uint i    = get_global_id(0);
    if(i >= ntidx)
        return;
    uint dx    = ((m & 1) != 0 ? 1 : ((m & 3) != 0 ? 2 : 4));
    uint s     = i < dimx ? i : (dimx - dx);
    indices[i] = ((s / m) * m * x + (s % m)) << 2;
}
__kernel void cl_fconv_generate_imap(__global uint* indices,
                                     uint snx,
                                     uint sny,
                                     uint snz,
                                     uint onx,
                                     uint ony,
                                     uint onz,
                                     uint inc,
                                     uint bat,
                                     uint su,
                                     uint sv,
                                     uint sd,
                                     uint ntidx)
{
    uint nvalid, npix, npix_uv, i;
    npix_uv = onx * ony;
    npix    = npix_uv * onz;
    nvalid  = bat * npix;
    i       = get_global_id(0);
    if(i >= ntidx)
        return;
    uint pix, uv, tid, value;
    tid   = i < nvalid ? i : (nvalid - 1);
    pix   = tid % npix;
    uv    = pix % npix_uv;
    value = ((((tid / npix) * snz * inc + sd * (pix / npix_uv)) * sny + sv * (uv / onx)) * snx +
             su * (uv % onx))
            << 2;
    indices[i] = value;
}
__kernel void cl_fconv_generate_omap(
    __global uint* indices, uint onx, uint ony, uint onz, uint onc, uint bat, uint ntidx)
{
    uint nvalid, npix, i, tid, value;

    npix   = onx * ony * onz;
    nvalid = bat * npix;
    i      = get_global_id(0);
    if(i >= ntidx)
        return;

    tid        = i < nvalid ? i : (nvalid - 1);
    value      = ((tid / npix) * npix * onc + (tid % npix)) << 2;
    indices[i] = value;
}

#define PSIZE(n, m) (((n) + (m)-1) & (~((m)-1)))
__kernel void cl_fconv_generate_span(__global uint* p_span,
                                     uint dnx,
                                     uint dny,
                                     uint dnz,
                                     uint fnx,
                                     uint fny,
                                     uint fnz,
                                     uint inc,
                                     uint du,
                                     uint dv,
                                     uint dd,
                                     uint izero)
{
    uint tid = get_global_id(0);
    uint x, y, z, c, n, pn, x8_left;
    int ldc, ldz;

    ldc = fnx * fny * fnz;
    ldz = fnx * fny;

    x         = tid % fnx;
    y         = (tid / fnx) % fny;
    z         = (tid / ldz) % fnz;
    c         = (tid / ldc) % inc;
    n         = ldc * inc;
    pn        = PSIZE(n, 8);
    x8_left   = pn - n;
    int total = n > 16 ? n : 16;

    if(tid >= total)
        return;

    p_span[c * ldc + z * ldz + y * fnx + x] = (((c * dnz + z * dd) * dny + y * dv) * dnx + x * du)
                                              << 2;

    p_span += n;
    if(x8_left)
    {
        if(tid < x8_left)
        {
            p_span[tid] = izero;
        }
    }
    p_span += x8_left;
    if(tid < 16)
    {
        p_span[tid] = 0;
    }
}
