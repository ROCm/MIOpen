/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

// NOTE: These included headers should be standalone, they shouldn't relay on each other,
// otherwise the dependencies will be pretty messed up.
// TODO: actually these headers are not that independent due to the Macros that being used
// currently. we should remove as more macros as we can.
#include "batchnorm_functions.hpp"
#include "bnorm_spatial_activation_functions.hpp"
#include "activation_functions.hpp"
#include "reduction_functions.hpp"
#include "static_unroll.hpp"

// Load the configs to this file
namespace /*anonymous*/ {
using mio_config    = miopen::config;
using mio_bn_config = miopen::batchnorm::config;
} // namespace

namespace miopen {
namespace batchnorm {

template <int MIoBnVariant, typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormFwdTrainSpatialHIPImpl
{
    static_assert(false, "this variant is not supported.");
};

// This is the instance for MIO_BN_VARIANT == 1
template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormFwdTrainSpatialHIPImpl<1, FpType, FpPrecType, FpAccumType>
{
    // These are the configs for this variant
    static constexpr unsigned int max_read =
        mio_config::layout_nhwc ? 1 : (mio_bn_config::hw >= 4096 ? 3 : 2);
    static constexpr unsigned int rd_blk = 1;
    static constexpr unsigned int grprd  = mio_config::layout_nhwc
                                               ? (mio_bn_config::launch_dim.grp0 * rd_blk)
                                               : (mio_bn_config::launch_dim.grp0 * rd_blk * 4);
    static constexpr unsigned int rem4 =
        mio_bn_config::nhw - ((mio_bn_config::nhw / grprd) * grprd);
    static constexpr unsigned int less4  = mio_bn_config::nhw - rem4;
    static constexpr unsigned int chunk4 = max_read * grprd;
    static constexpr unsigned int remout4 =
        mio_bn_config::nhw - ((mio_bn_config::nhw / chunk4) * chunk4);
    static constexpr unsigned int lessout4 = mio_bn_config::nhw - remout4;
    static constexpr unsigned int rem =
        mio_bn_config::nhw -
        ((mio_bn_config::nhw / mio_bn_config::launch_dim.grp0) * mio_bn_config::launch_dim.grp0);
    static constexpr unsigned int less  = mio_bn_config::nhw - rem;
    static constexpr unsigned int chunk = max_read * mio_bn_config::launch_dim.grp0;
    static constexpr unsigned int remout =
        mio_bn_config::nhw - ((mio_bn_config::nhw / chunk) * chunk);
    static constexpr unsigned int lessout = mio_bn_config::nhw - remout;

    // Kernel
    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict scale,
                                                         const FpPrecType* __restrict bias,
                                                         FpPrecType INHW,
                                                         double epsilon,
                                                         FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        FpPrecType pvscale, pvbias;

        mean        = 0;
        variance    = 0;
        invVariance = 0;

        __shared__ FpPrecType lcl_bias;
        __shared__ FpPrecType lcl_scale;

        unsigned int index       = 0;
        const unsigned int lid   = threadIdx.x;
        const unsigned int grpid = blockIdx.x;

        // Note: this variable is only used when mio_config::layout_nhwc is false.
        unsigned int chwid;
        if constexpr(!mio_config::layout_nhwc)
        {
            chwid = grpid * mio_bn_config::hw;
        }

        unsigned int nidx  = 0;
        unsigned int hwidx = 0;

        if(lid == 0)
        {
            lcl_scale = *(scale + grpid);
            lcl_bias  = *(bias + grpid);
        }

        __syncthreads();

        if constexpr(!mio_config::layout_nhwc && mio_bn_config::hw >= 4096)
        {
            using fp_type4 = typename mapped_vector_type<FpType, 4>::type;
            fp_type4 read4;

            static_unroll_count<unsigned int, 0, less4, grprd, 2>{[&](unsigned int k) {
                if((k + (lid << 2)) < less4)
                {
                    nidx  = (k + (lid << 2)) / mio_bn_config::hw;
                    hwidx = (k + (lid << 2)) - (nidx * mio_bn_config::hw);
                    index = nidx * mio_bn_config::chw + chwid + hwidx;
                    read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                    miopen::batchnorm::_accumulate4(mean, read4);
                    miopen::batchnorm::_accumulate_mad4(variance, read4, read4, variance);
                }
            }};

            if constexpr(rem4 > 0u)
            {
                const unsigned int remkey = (lid << 2) + less4;
                nidx                      = remkey / mio_bn_config::hw;
                hwidx                     = remkey - (nidx * mio_bn_config::hw);
                index                     = nidx * mio_bn_config::chw + chwid + hwidx;

                // TODO: mio_bn_config::nchw could smaller than 3, mio_bn_config::nchw - 3
                // may result in a negative result, index is unsigned int, so the value on
                // the right of `<` will decay to unsigned int, a negative value will become
                // a really large positive value.
                if(index < (mio_bn_config::nchw - 3))
                {
                    read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                    miopen::batchnorm::_accumulate4(mean, read4);
                    miopen::batchnorm::_accumulate_mad4(variance, read4, read4, variance);
                }
            }
        }
        else
        {
            static_unroll_count<unsigned int, 0, less, mio_bn_config::launch_dim.grp0, 4>{
                [&](unsigned int k) {
                    if(k + lid < less)
                    {
                        nidx  = (k + lid) / mio_bn_config::hw;
                        hwidx = (k + lid) - (nidx * mio_bn_config::hw);
                        if constexpr(mio_config::layout_nhwc)
                        {
                            index = nidx * mio_bn_config::chw + hwidx * mio_bn_config::c + grpid;
                        }
                        else
                        {
                            index = nidx * mio_bn_config::chw + chwid + hwidx;
                        }
                        const auto xin = cast<FpPrecType>(in[index]);
                        mean += xin;
                        variance = fma(xin, xin, variance);
                    }
                }};

            if constexpr(rem > 0u)
            {
                // Note: hip compiler has a bug, it throws compiler warning for comparing unsigned
                // int with 0 value, when rem is 0. but when rem is 0, this code block should not be
                // compiler due to we use if constexpr above.
                if(static_cast<int>(lid) < static_cast<int>(rem))
                {
                    unsigned int remkey = lid + less;
                    nidx                = remkey / mio_bn_config::hw;
                    hwidx               = remkey - (nidx * mio_bn_config::hw);
                    if constexpr(mio_config::layout_nhwc)
                    {
                        index = nidx * mio_bn_config::chw + hwidx * mio_bn_config::c + grpid;
                    }
                    else
                    {
                        index = nidx * mio_bn_config::chw + chwid + hwidx;
                    }

                    const auto xin =
                        index < mio_bn_config::nchw ? cast<FpPrecType>(in[index]) : FpPrecType{0};
                    mean += xin;
                    variance = fma(xin, xin, variance);
                }
            }
        }

        __syncthreads();

        constexpr auto lcl_data_size =
            mio_bn_config::use_amdgnc ? mio_bn_config::lds_gcn_size : mio_bn_config::lds_size;
        __shared__ FpAccumType lcl_data_x[lcl_data_size];
        __shared__ FpAccumType lcl_data_y[lcl_data_size];
        if constexpr(mio_bn_config::use_amdgnc)
        {
            miopen::reduction::gcn_reduce2<FpAccumType, lcl_data_size>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(INHW),
                lcl_data_x,
                lcl_data_y,
                lid);
        }
        else
        {
            miopen::reduction::lds_reduce2<FpAccumType, lcl_data_size>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(INHW),
                lcl_data_x,
                lcl_data_y,
                lid);
        }

        // REDUCTION COMPLETE ---------------------------
        variance = fma(-mean, mean, variance);
        if(variance < FpPrecType{0})
        {
            variance = FpPrecType{0};
        }

        // unsafe: casting double to FpPrecType
        invVariance = rsqrt(variance + static_cast<FpPrecType>(epsilon));
        pvscale     = lcl_scale;
        pvbias      = lcl_bias;
        if constexpr(mio_config::layout_nhwc || rem == 0)
        {
            constexpr unsigned int k_limit = mio_config::layout_nhwc ? mio_bn_config::nhw : less;

            static_unroll_count<unsigned int, 0, k_limit, mio_bn_config::launch_dim.grp0, 2>{
                [&](unsigned int k) {
                    if(k + lid < k_limit)
                    {
                        nidx  = (k + lid) / mio_bn_config::hw;
                        hwidx = (k + lid) - (nidx * mio_bn_config::hw);
                        if constexpr(mio_config::layout_nhwc)
                        {
                            index = nidx * mio_bn_config::chw + hwidx * mio_bn_config::c + grpid;
                        }
                        else
                        {
                            index = nidx * mio_bn_config::chw + chwid + hwidx;
                        }

                        out[index] = cast<FpType>(miopen::batchnorm::activation_op(
                            fma(pvscale,
                                (cast<FpPrecType>(in[index]) - mean) * invVariance,
                                pvbias),
                            alpha,
                            beta));
                    }
                }};
        }
        else
        {
            FpPrecType xhat[max_read];

            static_unroll_count<unsigned int, 0, lessout, chunk, 2>{[&](unsigned int k) {
                if(k + (max_read * lid) < lessout)
                {
                    for(unsigned int j = 0; j < max_read; ++j)
                    {
                        const unsigned int l = k + (max_read * lid) + j;
                        nidx                 = l / mio_bn_config::hw;
                        hwidx                = l - (nidx * mio_bn_config::hw);
                        index                = nidx * mio_bn_config::chw + chwid + hwidx;
                        xhat[j]              = (cast<FpPrecType>(in[index]) - mean) * invVariance;
                    }

                    __syncthreads();

                    for(unsigned int j = 0; j < max_read; ++j) // This part takes 0.405
                    {
                        const unsigned int l = k + (max_read * lid) + j;
                        nidx                 = l / mio_bn_config::hw;
                        hwidx                = l - (nidx * mio_bn_config::hw);
                        index                = nidx * mio_bn_config::chw + chwid + hwidx;
                        out[index]           = cast<FpType>(miopen::batchnorm::activation_op(
                            fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }};

            if constexpr(remout > 0u)
            {
                const unsigned int remkeyout = (max_read * lid) + lessout;
                for(unsigned int j = 0; j < max_read; ++j)
                {
                    unsigned int l = remkeyout + j;
                    nidx           = l / mio_bn_config::hw;
                    hwidx          = l - (nidx * mio_bn_config::hw);
                    index          = nidx * mio_bn_config::chw + chwid + hwidx;
                    // TODO: comparing different types
                    const auto xin =
                        (index < mio_bn_config::nchw) ? cast<FpPrecType>(in[index]) : FpPrecType{0};
                    xhat[j] = (xin - cast<FpPrecType>(mean)) * cast<FpPrecType>(invVariance);
                }

                __syncthreads();
                for(unsigned int j = 0; j < max_read; ++j)
                {
                    const unsigned int l = remkeyout + j;
                    nidx                 = l / mio_bn_config::hw;
                    hwidx                = l - (nidx * mio_bn_config::hw);
                    index                = nidx * mio_bn_config::chw + chwid + hwidx;

                    if(index < mio_bn_config::nchw)
                    {
                        out[index] = cast<FpType>(miopen::batchnorm::activation_op(
                            fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }
        }
        return;
    }
};

} // namespace batchnorm
} // namespace miopen

/// C interfaces

// TODO: This can be removed after every variant has been implemnted
// [[deprecated]]
#if(MIO_BN_VARIANT != 2)
extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormFwdTrainSpatialHIP(
        const typename mio_bn_config::fp_type* __restrict in,
        typename mio_bn_config::fp_type* __restrict out,
        const typename mio_bn_config::fp_prec_type* __restrict scale,
        const typename mio_bn_config::fp_prec_type* __restrict bias,
        typename mio_bn_config::fp_prec_type INHW,
// TODO: should find a better way of doing this
// but it's hard becasue C does not support function
// overloads.
// [[deprecated]]
#if(MIO_RUNNING_RESULT == 1)
        double expAvgFactor,
        typename mio_bn_config::fp_prec_type* __restrict resultRunningMean,
        typename mio_bn_config::fp_prec_type* __restrict resultRunningVariance,
#endif
        double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        ,
        typename mio_bn_config::fp_prec_type* __restrict resultSaveMean,
        typename mio_bn_config::fp_prec_type* __restrict resultSaveInvVariance
#endif
        ,
        typename mio_bn_config::fp_prec_type alpha,
        typename mio_bn_config::fp_prec_type beta)
{
    using fp_type          = typename mio_bn_config::fp_type;
    using fp_prec_type     = typename mio_bn_config::fp_prec_type;
    using fp_accum_type    = typename mio_bn_config::fp_accum_type;
    using fp_accum_c_type  = typename mio_bn_config::fp_accum_c_type;
    using fp_prec_c_type   = typename mio_bn_config::fp_prec_c_type;
    constexpr auto variant = mio_bn_config::variant;

    using forward_train_spatial_impl = miopen::batchnorm::
        MIOpenBatchNormFwdTrainSpatialHIPImpl<variant, fp_type, fp_prec_type, fp_accum_type>;

    fp_prec_type mean, variance, invVariance;
    const unsigned int lid   = threadIdx.x;
    const unsigned int grpid = blockIdx.x;

    forward_train_spatial_impl{}(
        in, out, scale, bias, INHW, epsilon, mean, variance, invVariance, alpha, beta);

    if(lid == 0)
    {
// TODO: this should also be removed, but using constexpr can lead compile error
#if(MIO_RUNNING_RESULT == 1)
        miopen::batchnorm::running_stash<fp_accum_type, fp_accum_c_type, fp_prec_c_type>(
            resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, grpid);
#endif
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        miopen::batchnorm::saved_stash<fp_accum_c_type, fp_prec_c_type>(
            resultSaveMean, resultSaveInvVariance, mean, invVariance, grpid);
#endif
    }
}

#endif
