#pragma once
#include "host_tensor.hpp"

template <typename TIn,
          typename TWei,
          typename TOut,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void host_direct_convolution_backward_data(Tensor<TIn>& in,
                                           const Tensor<TWei>& wei,
                                           const Tensor<TOut>& out,
                                           const ConvStrides& conv_strides,
                                           const ConvDilations& conv_dilations,
                                           const InLeftPads& in_left_pads,
                                           const InRightPads& in_right_pads,
                                           const ConvTensorLayout layout = ConvTensorLayout::NCHW)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    auto f_nchw = [&](auto n, auto c, auto hi, auto wi) {
        std::size_t N  = in.mDesc.GetLengths()[I0];
        std::size_t C  = in.mDesc.GetLengths()[I1];
        std::size_t Hi = in.mDesc.GetLengths()[I2];
        std::size_t Wi = in.mDesc.GetLengths()[I3];

        std::size_t K = wei.mDesc.GetLengths()[I0];
        std::size_t Y = wei.mDesc.GetLengths()[I2];
        std::size_t X = wei.mDesc.GetLengths()[I3];

        std::size_t Ho = out.mDesc.GetLengths()[I2];
        std::size_t Wo = out.mDesc.GetLengths()[I3];

        double v = 0;

        for(int y = 0; y < Y; ++y)
        {
            int h_tmp = hi + in_left_pads[I0] - y * conv_dilations[I0];

            if(h_tmp % conv_strides[I0] == 0)
            {
                int ho = h_tmp / conv_strides[I0];

                if(ho >= 0 && ho < Ho)
                {
                    for(int x = 0; x < X; ++x)
                    {
                        int w_tmp = wi + in_left_pads[I1] - x * conv_dilations[I1];

                        if(w_tmp % conv_strides[I1] == 0)
                        {
                            int wo = w_tmp / conv_strides[I1];

                            if(wo >= 0 && wo < Wo)
                            {
                                for(int k = 0; k < K; ++k)
                                {
                                    v += out(n, k, ho, wo) * wei(k, c, y, x);
                                }
                            }
                        }
                    }
                }
            }
        }

        in(n, c, hi, wi) = v;
    };

    auto f_nhwc = [&](auto n, auto hi, auto wi, auto c) {
        std::size_t N  = in.mDesc.GetLengths()[I0];
        std::size_t Hi = in.mDesc.GetLengths()[I1];
        std::size_t Wi = in.mDesc.GetLengths()[I2];
        std::size_t C  = in.mDesc.GetLengths()[I3];

        std::size_t K = wei.mDesc.GetLengths()[I0];
        std::size_t Y = wei.mDesc.GetLengths()[I1];
        std::size_t X = wei.mDesc.GetLengths()[I2];

        std::size_t Ho = out.mDesc.GetLengths()[I1];
        std::size_t Wo = out.mDesc.GetLengths()[I2];

        double v = 0;

        for(int y = 0; y < Y; ++y)
        {
            int h_tmp = hi + in_left_pads[I0] - y * conv_dilations[I0];

            if(h_tmp % conv_strides[I0] == 0)
            {
                int ho = h_tmp / conv_strides[I0];

                if(ho >= 0 && ho < Ho)
                {
                    for(int x = 0; x < X; ++x)
                    {
                        int w_tmp = wi + in_left_pads[I1] - x * conv_dilations[I1];

                        if(w_tmp % conv_strides[I1] == 0)
                        {
                            int wo = w_tmp / conv_strides[I1];

                            if(wo >= 0 && wo < Wo)
                            {
                                for(int k = 0; k < K; ++k)
                                {
                                    v += out(n, ho, wo, k) * wei(k, y, x, c);
                                }
                            }
                        }
                    }
                }
            }
        }

        in(n, hi, wi, c) = v;
    };

    switch(layout)
    {
    case ConvTensorLayout::NCHW:
        make_ParallelTensorFunctor(f_nchw,
                                   in.mDesc.GetLengths()[0],
                                   in.mDesc.GetLengths()[1],
                                   in.mDesc.GetLengths()[2],
                                   in.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
        break;
    case ConvTensorLayout::NHWC:
        make_ParallelTensorFunctor(f_nhwc,
                                   in.mDesc.GetLengths()[0],
                                   in.mDesc.GetLengths()[1],
                                   in.mDesc.GetLengths()[2],
                                   in.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
        break;
    default: throw std::runtime_error("wrong! not supported layout");
    }
}
