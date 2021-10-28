#pragma once
#include "host_tensor.hpp"

template <typename TOut,
          typename TIn,
          typename TWei,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_direct_convolution_backward_weights(
    const Tensor<TOut>& out,
    const Tensor<TIn>& in,
    Tensor<TWei>& wei,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads&,
    const ConvTensorLayout layout = ConvTensorLayout::NCHW)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    auto f_kcyx       = [&](auto k, auto c, auto y, auto x) {
        double v = 0;
        for(int n = 0; n < out.mDesc.GetLengths()[0]; ++n)
        {
            for(int ho = 0; ho < out.mDesc.GetLengths()[2]; ++ho)
            {
                int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                for(int wo = 0; wo < out.mDesc.GetLengths()[3]; ++wo)
                {
                    int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[3])
                    {
                        v += static_cast<const double>(in(n, c, hi, wi)) *
                             static_cast<const double>(out(n, k, ho, wo));
                    }
                }
            }
        }
        wei(k, c, y, x) = v;
    };

    auto f_kyxc = [&](auto k, auto y, auto x, auto c) {
        double v = 0;
        for(int n = 0; n < out.mDesc.GetLengths()[0]; ++n)
        {
            for(int ho = 0; ho < out.mDesc.GetLengths()[1]; ++ho)
            {
                int hi = ho * conv_strides[I0] + y * conv_dilations[I0] - in_left_pads[I0];
                for(int wo = 0; wo < out.mDesc.GetLengths()[2]; ++wo)
                {
                    int wi = wo * conv_strides[I1] + x * conv_dilations[I1] - in_left_pads[I1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[1] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[2])
                    {
                        v += static_cast<const double>(in(n, hi, wi, c)) *
                             static_cast<const double>(out(n, ho, wo, k));
                    }
                }
            }
        }
        wei(k, y, x, c) = v;
    };

    if(layout == ConvTensorLayout::NCHW)
    {
        make_ParallelTensorFunctor(f_kcyx,
                                   wei.mDesc.GetLengths()[0],
                                   wei.mDesc.GetLengths()[1],
                                   wei.mDesc.GetLengths()[2],
                                   wei.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
    }
    else if(layout == ConvTensorLayout::NHWC)
    {
        make_ParallelTensorFunctor(f_kyxc,
                                   wei.mDesc.GetLengths()[0],
                                   wei.mDesc.GetLengths()[1],
                                   wei.mDesc.GetLengths()[2],
                                   wei.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
    }
    else
    {
        throw std::runtime_error("wrong! not supported layout");
    }
}
