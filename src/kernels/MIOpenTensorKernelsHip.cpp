/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

__device__ inline MIOPEN_TYPE miopenAdd(MIOPEN_TYPE a, MIOPEN_TYPE b) { return a + b; }

__device__ inline MIOPEN_TYPE miopenMul(MIOPEN_TYPE a, MIOPEN_TYPE b) { return a * b; }

__device__ inline MIOPEN_TYPE miopenMax(MIOPEN_TYPE a, MIOPEN_TYPE b) { return ((a > b) ? a : b); }

__device__ inline MIOPEN_TYPE miopenMin(MIOPEN_TYPE a, MIOPEN_TYPE b) { return ((a < b) ? a : b); }

// NCH
extern "C" __global__ void Op3dTensorGeneric(const MIOPEN_TYPE* __restrict__ a,
                                             const int a_nstride,
                                             const int a_cstride,
                                             const int a_hstride,
                                             const MIOPEN_TYPE* __restrict__ b,
                                             const int b_nstride,
                                             const int b_cstride,
                                             const int b_hstride,
                                             MIOPEN_TYPE* __restrict__ c,
                                             const int c_c,
                                             const int c_h,
                                             const int c_nstride,
                                             const int c_cstride,
                                             const int c_hstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const unsigned int bitmap,
                                             const long Aoffset,
                                             const long Boffset,
                                             const long Coffset,
                                             const int total_work,
                                             const int use_beta)
{
    const MIOPEN_TYPE* __restrict__ a_off = a + Aoffset;
    const MIOPEN_TYPE* __restrict__ b_off = b + Boffset;
    MIOPEN_TYPE* __restrict__ c_off       = c + Coffset;

    const int b_nstride_res = b_nstride * ((bitmap >> 2) & 1);
    const int b_cstride_res = b_cstride * ((bitmap >> 1) & 1);
    const int b_hstride_res = b_hstride * ((bitmap >> 0) & 1);

    for(int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total_work;
        gid += gridDim.x * blockDim.x)
    {
        int o_h = gid % c_h;
        int o_c = (gid / c_h) % c_c;
        int o_n = (gid / c_h) / c_c;

        int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride;
        int bindex = o_n * b_nstride_res + o_c * b_cstride_res + o_h * b_hstride_res;
        int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride;

        MIOPEN_TYPE res = MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, b_off[bindex] * alpha1);
        c_off[cindex]   = use_beta == 1 ? c_off[cindex] * beta + res : res;
    }
}
