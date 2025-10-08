// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bfloat16_dev.hpp"

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#ifndef MIOPEN_USE_FPMIX
#define MIOPEN_USE_FPMIX 0
#endif

#ifndef MIOPEN_USE_BFPMIX
#define MIOPEN_USE_BFPMIX 0
#endif

#define _FLOAT_ACCUM float
#if MIOPEN_USE_FP16 == 1
#define MIO_BN_NODPP 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_PREC half
#define EPSILON (_FLOAT)0.0001
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define _FLOAT_PREC float
#define EPSILON (_FLOAT)0.000001
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif
#if MIOPEN_USE_FPMIX == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half

#ifdef MIO_BN_NODPP
#undef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#endif

#ifdef _FLOAT_PREC
#undef _FLOAT_PREC
#endif
#define _FLOAT_PREC float

#ifdef EPSILON
#undef EPSILON
#endif
#define EPSILON (_FLOAT)0.000001

#endif

#if MIOPEN_USE_BFPMIX == 1
#define _FLOAT ushort

#ifdef MIO_BN_NODPP
#undef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#endif

#ifdef _FLOAT_PREC
#undef _FLOAT_PREC
#endif
#define _FLOAT_PREC float

#ifdef EPSILON
#undef EPSILON
#endif
#define EPSILON (_FLOAT_PREC)0.000001

#define FLOAT2FLOATPREC(x) (bfloat16_to_float(x))
#define FLOATPREC2FLOAT(x) (float_to_bfloat16(x))
#define FLOAT2ACCUM(x) (FLOAT2FLOATPREC(x))
#define ACCUM2FLOAT(x) (FLOATPREC2FLOAT(x))
#define ACCUM2FLOATPREC(x) (x)

#else

#define FLOAT2FLOATPREC(x) ((_FLOAT_PREC)(x))
#define FLOATPREC2FLOAT(x) ((_FLOAT)(x))
#define FLOAT2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define ACCUM2FLOAT(x) ((_FLOAT)(x))
#define ACCUM2FLOATPREC(x) ((_FLOAT_PREC)(x))
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)
#define _AS_FLOAT PPCAT(as_, _FLOAT)

#define _FLOAT_PREC2 PPCAT(_FLOAT_PREC, TWO)
#define _FLOAT_ACCUM2 PPCAT(_FLOAT_ACCUM, TWO)
#define _FLOAT_PREC4 PPCAT(_FLOAT_PREC, FOUR)
#define _FLOAT_ACCUM4 PPCAT(_FLOAT_ACCUM, FOUR)
#define _FLOAT_PREC8 PPCAT(_FLOAT_PREC, EIGHT)
#define _FLOAT_ACCUM8 PPCAT(_FLOAT_ACCUM, EIGHT)

#ifndef MIO_BN_LDSGCN_SIZE
#define MIO_BN_LDSGCN_SIZE 16
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 256
#endif

#ifndef MIO_BN_C
#define MIO_BN_C 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#ifndef MIO_BN_GRP0
#define MIO_BN_GRP0 1
#endif

#ifndef MIO_BN_GRP1
#define MIO_BN_GRP1 1
#endif

#ifndef MIO_BN_GRP2
#define MIO_BN_GRP2 1
#endif

#ifndef MIO_BN_NGRPS
#define MIO_BN_NGRPS 1
#endif

#ifndef MIO_BN_LOOP_UNROLL_MAXN
#define MIO_BN_LOOP_UNROLL_MAXN 768
#endif

#ifndef MIO_BN_LOOP_UNROLL_MAXHW
#define MIO_BN_LOOP_UNROLL_MAXHW 2500
#endif

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 255
#endif

#ifndef MIO_BN_MAXN
#define MIO_BN_MAXN 65
#endif

// TODO: Spaghetti code!!!
// MIOPEN_USE_AMDGCN may be defined before this header.
#ifndef MIOPEN_USE_AMDGCN
#if defined(__AMDGCN__) &&                           \
    !((defined(MIO_BN_GFX103X) && MIO_BN_GFX103X) || \
      (defined(MIO_BN_GFX110X) && MIO_BN_GFX110X) || \
      (defined(MIO_BN_GFX120X) && MIO_BN_GFX120X) || \
      (defined(MIO_BN_GFX115X) && MIO_BN_GFX115X))
#define MIOPEN_USE_AMDGCN 1
#else
#define MIOPEN_USE_AMDGCN 0
#endif
#endif

// MIOPEN_USE_AMDGCN is guaranteed to be defined at this point.

#ifndef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#elif(MIO_BN_NODPP == 1 && MIO_BN_VARIANT != 0)
#undef MIOPEN_USE_AMDGCN
#define MIOPEN_USE_AMDGCN 0
#endif

#ifndef MIO_SAVE_MEAN_VARIANCE
#define MIO_SAVE_MEAN_VARIANCE 0
#endif

#ifndef MIO_RUNNING_RESULT
#define MIO_RUNNING_RESULT 0
#endif

#ifndef MIO_BN_GFX103X
#define MIO_BN_GFX103X 0
#endif

#ifndef MIO_BN_GFX110X
#define MIO_BN_GFX110X 0
#endif

#ifndef MIO_BN_GFX120X
#define MIO_BN_GFX120X 0
#endif

#ifndef MIO_BN_GFX115X
#define MIO_BN_GFX115X 0
#endif

#ifndef MIO_BN_VECTORIZE
#define MIO_BN_VECTORIZE 0
#endif

#ifndef MIO_BN_VEC_SIZE
#define MIO_BN_VEC_SIZE 1
#endif

#ifndef MIO_BN_STASH_METHOD
#define MIO_BN_STASH_METHOD 0
#endif

#define FLOATPREC4_2_FLOAT4(val)       \
    ((_FLOAT4)(FLOATPREC2FLOAT(val.x), \
               FLOATPREC2FLOAT(val.y), \
               FLOATPREC2FLOAT(val.z), \
               FLOATPREC2FLOAT(val.w)))

#define FLOATPREC2_2_FLOAT2(val) ((_FLOAT2)(FLOATPREC2FLOAT(val.x), FLOATPREC2FLOAT(val.y)))

#define FLOATPREC8_2_FLOAT8(val)        \
    ((_FLOAT8)(FLOATPREC2FLOAT(val.s0), \
               FLOATPREC2FLOAT(val.s1), \
               FLOATPREC2FLOAT(val.s2), \
               FLOATPREC2FLOAT(val.s3), \
               FLOATPREC2FLOAT(val.s4), \
               FLOATPREC2FLOAT(val.s5), \
               FLOATPREC2FLOAT(val.s6), \
               FLOATPREC2FLOAT(val.s7)))

#define FLOAT4_2_FLOATPREC4(val)            \
    ((_FLOAT_PREC4)(FLOAT2FLOATPREC(val.x), \
                    FLOAT2FLOATPREC(val.y), \
                    FLOAT2FLOATPREC(val.z), \
                    FLOAT2FLOATPREC(val.w)))

#define FLOAT8_2_FLOATPREC8(val)             \
    ((_FLOAT_PREC8)(FLOAT2FLOATPREC(val.s0), \
                    FLOAT2FLOATPREC(val.s1), \
                    FLOAT2FLOATPREC(val.s2), \
                    FLOAT2FLOATPREC(val.s3), \
                    FLOAT2FLOATPREC(val.s4), \
                    FLOAT2FLOATPREC(val.s5), \
                    FLOAT2FLOATPREC(val.s6), \
                    FLOAT2FLOATPREC(val.s7)))

#define FLOAT2_2_FLOATPREC2(val) ((_FLOAT_PREC2)(FLOAT2FLOATPREC(val.x), FLOAT2FLOATPREC(val.y)))

#define _ACCUMULATE1(a, b) a += b;

#define _ACCUMULATE_MAD1(a, b, c, d) a = mad(b, c, d);

#define _ACCUMULATE4(a, b) \
    a += b.x;              \
    a += b.y;              \
    a += b.z;              \
    a += b.w;

#define _ACCUMULATE8(a, b) \
    a += b.s0;             \
    a += b.s1;             \
    a += b.s2;             \
    a += b.s3;             \
    a += b.s4;             \
    a += b.s5;             \
    a += b.s6;             \
    a += b.s7;

#define _ACCUMULATE2(a, b) \
    a += b.x;              \
    a += b.y;

#define _ACCUMULATE_MAD4(a, b, c, d) \
    a = mad(b.x, c.x, d);            \
    a = mad(b.y, c.y, d);            \
    a = mad(b.z, c.z, d);            \
    a = mad(b.w, c.w, d);

#define _ACCUMULATE_MAD8(a, b, c, d) \
    a = mad(b.s0, c.s0, d);          \
    a = mad(b.s1, c.s1, d);          \
    a = mad(b.s2, c.s2, d);          \
    a = mad(b.s3, c.s3, d);          \
    a = mad(b.s4, c.s4, d);          \
    a = mad(b.s5, c.s5, d);          \
    a = mad(b.s6, c.s6, d);          \
    a = mad(b.s7, c.s7, d);

#define _ACCUMULATE_MAD2(a, b, c, d) \
    a = mad(b.x, c.x, d);            \
    a = mad(b.y, c.y, d);

#if MIO_BN_VECTORIZE

#if MIO_BN_VEC_SIZE == 4
// Case vectorsize 4
#if MIO_LAYOUT_NHWC
// NHWC vectorize in X direction which corresponds
// to channels
#define VEC_SIZE_X MIO_BN_VEC_SIZE
#define VEC_SIZE_Y 1
// _C suffix means used for computation
// _LS suffix means used for loading / storing
#define _FLOAT_PREC_C _FLOAT_PREC4
#define _FLOAT_PREC_LS _FLOAT_PREC4
#define _FLOAT_C _FLOAT4
#define _FLOAT_LS _FLOAT4
#define _FLOAT_ACCUM_C _FLOAT_ACCUM4
#define _FLOAT_ACCUM_LS _FLOAT_ACCUM4
#define _ACCUMULATE _ACCUMULATE1
#define _ACCUMULATE_MAD _ACCUMULATE_MAD1
#else
// NCHW vectorize in Y direction which corresponds
// to HW
#define VEC_SIZE_X 1
#define VEC_SIZE_Y MIO_BN_VEC_SIZE
#define _FLOAT_PREC_C _FLOAT_PREC
#define _FLOAT_PREC_LS _FLOAT_PREC4
// _C suffix means used for computation
// _LS suffix means used for loading / storing
#define _FLOAT_C _FLOAT
#define _FLOAT_LS _FLOAT4
#define _FLOAT_ACCUM_C _FLOAT_ACCUM
#define _FLOAT_ACCUM_LS _FLOAT_ACCUM4
#define _ACCUMULATE _ACCUMULATE4
#define _ACCUMULATE_MAD _ACCUMULATE_MAD4
#endif

#define FLOAT2FLOATPREC_VEC FLOAT4_2_FLOATPREC4
#define FLOATPREC2FLOAT_VEC FLOATPREC4_2_FLOAT4

#elif MIO_BN_VEC_SIZE == 8

// Case vectorsize 8
#if MIO_LAYOUT_NHWC
// NHWC vectorize in X direction which corresponds
// to channels
#define VEC_SIZE_X MIO_BN_VEC_SIZE
#define VEC_SIZE_Y 1
// _C suffix means used for computation
// _LS suffix means used for loading / storing
#define _FLOAT_PREC_C _FLOAT_PREC8
#define _FLOAT_PREC_LS _FLOAT_PREC8
#define _FLOAT_C _FLOAT8
#define _FLOAT_LS _FLOAT8
#define _FLOAT_ACCUM_C _FLOAT_ACCUM8
#define _FLOAT_ACCUM_LS _FLOAT_ACCUM8
#define _ACCUMULATE _ACCUMULATE1
#define _ACCUMULATE_MAD _ACCUMULATE_MAD1
#else
// NCHW vectorize in Y direction which corresponds
// to HW
#define VEC_SIZE_X 1
#define VEC_SIZE_Y MIO_BN_VEC_SIZE
#define _FLOAT_PREC_C _FLOAT_PREC
#define _FLOAT_PREC_LS _FLOAT_PREC8
// _C suffix means used for computation
// _LS suffix means used for loading / storing
#define _FLOAT_C _FLOAT
#define _FLOAT_LS _FLOAT8
#define _FLOAT_ACCUM_C _FLOAT_ACCUM
#define _FLOAT_ACCUM_LS _FLOAT_ACCUM8
#define _ACCUMULATE _ACCUMULATE8
#define _ACCUMULATE_MAD _ACCUMULATE_MAD8
#endif

#define FLOAT2FLOATPREC_VEC FLOAT8_2_FLOATPREC8
#define FLOATPREC2FLOAT_VEC FLOATPREC8_2_FLOAT8

#elif MIO_BN_VEC_SIZE == 2
// Case vectorsize 2
#if MIO_LAYOUT_NHWC
// NHWC vectorize in X direction which corresponds
// to channels
#define VEC_SIZE_X MIO_BN_VEC_SIZE
#define VEC_SIZE_Y 1
// _C suffix means used for computation
// _LS suffix means used for loading / storing
#define _FLOAT_PREC_C _FLOAT_PREC2
#define _FLOAT_PREC_LS _FLOAT_PREC2
#define _FLOAT_C _FLOAT2
#define _FLOAT_LS _FLOAT2
#define _FLOAT_ACCUM_C _FLOAT_ACCUM2
#define _FLOAT_ACCUM_LS _FLOAT_ACCUM2
#define _ACCUMULATE _ACCUMULATE1
#define _ACCUMULATE_MAD _ACCUMULATE_MAD1
#else
// NCHW vectorize in Y direction which corresponds
// to HW
#define VEC_SIZE_X 1
#define VEC_SIZE_Y MIO_BN_VEC_SIZE
#define _FLOAT_PREC_C _FLOAT_PREC
#define _FLOAT_PREC_LS _FLOAT_PREC2
// _C suffix means used for computation
// _LS suffix means used for loading / storing
#define _FLOAT_C _FLOAT
#define _FLOAT_LS _FLOAT2
#define _FLOAT_ACCUM_C _FLOAT_ACCUM
#define _FLOAT_ACCUM_LS _FLOAT_ACCUM4
#define _ACCUMULATE _ACCUMULATE2
#define _ACCUMULATE_MAD _ACCUMULATE_MAD2
#endif

#define FLOAT2FLOATPREC_VEC FLOAT2_2_FLOATPREC2
#define FLOATPREC2FLOAT_VEC FLOATPREC2_2_FLOAT2

#endif

#else
// Case vectorsize 1 (no vectorization)
#define VEC_SIZE 1
#define VEC_SIZE_X 1
#define VEC_SIZE_Y 1
#define _FLOAT_PREC_C _FLOAT_PREC
#define _FLOAT_PREC_LS _FLOAT_PREC
#define _FLOAT_C _FLOAT
#define _FLOAT_LS _FLOAT
#define _FLOAT_ACCUM_C _FLOAT_ACCUM
#define _FLOAT_ACCUM_LS _FLOAT_ACCUM
#define FLOAT2FLOATPREC_VEC FLOAT2FLOATPREC
#define FLOATPREC2FLOAT_VEC FLOATPREC2FLOAT
#define _ACCUMULATE _ACCUMULATE1
#define _ACCUMULATE_MAD _ACCUMULATE_MAD1

#endif

#define UNUSED __attribute__((__unused__))

#if(MIO_BN_VARIANT == 2)

#if(MIO_BN_STASH_METHOD == 0)
// store values in HW dimension
#define NSTRIDE ystride
#else
// store values in N dimension
#define NSTRIDE (MIO_BN_C / VEC_SIZE_X * MIO_BN_HW)
#endif

inline unsigned int getStashIndex(unsigned int vindex,
                                  unsigned int zgroupoffset,
                                  unsigned int ygroupoffset,
                                  unsigned int ystride,
                                  unsigned int xgrp_sz,
                                  unsigned int xgrp_id,
                                  unsigned int xlid,
                                  unsigned int xstride)
{
#if MIOPEN_USE_FPMIX || MIOPEN_USE_BFPMIX
    // 2 _FLOAT values are used to store 1 _FLOAT_PREC value.
#if MIO_LAYOUT_NHWC
#if MIO_BN_C % 2 == 0
    // xgrp_sz values are split in two parts: even threads use 2 values at even rows, odd threads -
    // at odd rows.
    // The only restriction for C and xgrp_sz is that they must be even.
    return zgroupoffset * (MIO_BN_C / VEC_SIZE_X * MIO_BN_HW) + (vindex * 2 + xlid % 2) * NSTRIDE +
           ygroupoffset * ystride + (xgrp_sz * xgrp_id + xlid / 2 * 2) * xstride;
#else
    // Values are stored consecutively in y dim.
    return zgroupoffset * (MIO_BN_C / VEC_SIZE_X * MIO_BN_HW) + (vindex * 2) * NSTRIDE +
           ygroupoffset * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
#endif
#else // !MIO_LAYOUT_NHWC
    // Values are stored consecutively in y dim, indices are aligned up by 2 (_FLOAT_PREC).
    return zgroupoffset * (MIO_BN_C / VEC_SIZE_X * MIO_BN_HW) +
           ((vindex * 2) * NSTRIDE + ygroupoffset * ystride + (xgrp_sz * xgrp_id + xlid) * xstride +
            1) /
               2 * 2;
#endif
#else
    return zgroupoffset * (MIO_BN_C / VEC_SIZE_X * MIO_BN_HW) + vindex * NSTRIDE +
           ygroupoffset * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
#endif
}

inline _FLOAT_PREC_C loadFromStash(const __global _FLOAT_C* stash,
                                   unsigned int vindex,
                                   unsigned int zgroupoffset,
                                   unsigned int ygroupoffset,
                                   unsigned int ystride,
                                   unsigned int xgrp_sz,
                                   unsigned int xgrp_id,
                                   unsigned int xlid,
                                   unsigned int xstride)
{
    unsigned int index =
        getStashIndex(vindex, zgroupoffset, ygroupoffset, ystride, xgrp_sz, xgrp_id, xlid, xstride);

#if(MIO_BN_STASH_METHOD == 0 || MIO_BN_STASH_METHOD == 1)
    return *((const __global _FLOAT_PREC_C*)(stash + index));
#else
    _FLOAT_PREC_C value;
    *((_FLOAT_C*)(&value)) = *(stash + index);
    index += NSTRIDE;
    *((_FLOAT_C*)(&value) + 1) = *(stash + index);

    return value;
#endif
}

inline void storeToStash(_FLOAT_PREC_C value,
                         __global _FLOAT_C* stash,
                         unsigned int vindex,
                         unsigned int zgroupoffset,
                         unsigned int ygroupoffset,
                         unsigned int ystride,
                         unsigned int xgrp_sz,
                         unsigned int xgrp_id,
                         unsigned int xlid,
                         unsigned int xstride)
{
    unsigned int index =
        getStashIndex(vindex, zgroupoffset, ygroupoffset, ystride, xgrp_sz, xgrp_id, xlid, xstride);

#if(MIO_BN_STASH_METHOD == 0 || MIO_BN_STASH_METHOD == 1)
    *((__global _FLOAT_PREC_C*)(stash + index)) = value;
#else
    *(stash + index) = *((_FLOAT_C*)(&value));
    index += NSTRIDE;
    *(stash + index) = *((_FLOAT_C*)(&value) + 1);
#endif
}
#endif

#if(MIO_BN_VARIANT != 4)
static inline void running_stash(global _FLOAT_PREC_C* resultRunningMean,
                                 global _FLOAT_PREC_C* resultRunningVariance,
                                 double expAvgFactor,
                                 _FLOAT_ACCUM_C mean,
                                 _FLOAT_ACCUM_C variance,
                                 uint channel)
{
    _FLOAT_ACCUM_C pvt_runMean = (_FLOAT_ACCUM_C)(*(resultRunningMean + channel));
    _FLOAT_ACCUM_C pvt_newRunMean =
        mad((_FLOAT_ACCUM)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
    resultRunningMean[channel] = (_FLOAT_PREC_C)mad(
        mean, (_FLOAT_ACCUM)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
    const _FLOAT_ACCUM_C adjust =
        (_FLOAT_ACCUM_C)((MIO_BN_NHW == 1)
                             ? variance
                             : variance * ((_FLOAT_ACCUM)MIO_BN_NHW /
                                           ((_FLOAT_ACCUM)MIO_BN_NHW - (_FLOAT_ACCUM)1.0)));
    resultRunningVariance[channel] =
        (_FLOAT_PREC_C)((1 - (_FLOAT_ACCUM)expAvgFactor) *
                            (_FLOAT_ACCUM_C)(*(resultRunningVariance + channel)) +
                        (_FLOAT_ACCUM)expAvgFactor * adjust);
}

static inline void running_stash_pa(global _FLOAT_PREC* resultRunningMean,
                                    global _FLOAT_PREC* resultRunningVariance,
                                    double expAvgFactor,
                                    _FLOAT_ACCUM mean,
                                    _FLOAT_ACCUM variance,
                                    uint index)
{
    _FLOAT_PREC N              = (_FLOAT_PREC)MIO_BN_N;
    _FLOAT_PREC pvt_runMean    = *(resultRunningMean + index); // previous: oldRunMean
    _FLOAT_PREC pvt_newRunMean = mad((_FLOAT_PREC)-expAvgFactor,
                                     pvt_runMean,
                                     pvt_runMean); // tmp = oldRunMean*(1-factor)

    resultRunningMean[index] = mad((_FLOAT_PREC)mean,
                                   (_FLOAT_PREC)expAvgFactor,
                                   pvt_newRunMean); // newMean*factor + tmp

    const _FLOAT_PREC adjust = (MIO_BN_N == 1) ? variance : variance * (N / (N - 1.0));
    resultRunningVariance[index] =
        (1 - (_FLOAT_PREC)expAvgFactor) * *(resultRunningVariance + index) +
        (_FLOAT_PREC)expAvgFactor * adjust;
}

#else

static inline void running_stash_dyn(global _FLOAT_PREC* resultRunningMean,
                                     global _FLOAT_PREC* resultRunningVariance,
                                     double expAvgFactor,
                                     _FLOAT_ACCUM mean,
                                     _FLOAT_ACCUM variance,
                                     uint channel,
                                     _FLOAT_ACCUM inhw)
{
    _FLOAT_ACCUM pvt_runMean = (_FLOAT_ACCUM)(*(resultRunningMean + channel));
    _FLOAT_ACCUM pvt_newRunMean =
        mad((_FLOAT_ACCUM)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
    resultRunningMean[channel] =
        (_FLOAT_PREC)mad(mean, (_FLOAT_ACCUM)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
    const _FLOAT_ACCUM adjust =
        (_FLOAT_ACCUM)((inhw == 1) ? variance : variance * (1. / (1. - inhw)));
    resultRunningVariance[channel] =
        (_FLOAT_PREC)((1 - (_FLOAT_ACCUM)expAvgFactor) *
                          (_FLOAT_ACCUM)(*(resultRunningVariance + channel)) +
                      (_FLOAT_ACCUM)expAvgFactor * adjust);
}
#endif

static inline void saved_stash(global _FLOAT_PREC_C* resultSaveMean,
                               global _FLOAT_PREC_C* resultSaveInvVariance,
                               _FLOAT_ACCUM_C mean,
                               _FLOAT_ACCUM_C invVariance,
                               uint channel)
{
    *(resultSaveMean + channel)        = (_FLOAT_PREC_C)mean;
    *(resultSaveInvVariance + channel) = (_FLOAT_PREC_C)invVariance;
}
