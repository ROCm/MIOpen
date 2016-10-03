#include <mlopen.h>
#include "test.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <mlopen/tensor.hpp>
#include <mlopen/convolution.hpp>
#include <limits>

#include "network_data.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

template<class T>
tensor<T> get_output_tensor(const mlopen::ConvolutionDescriptor& filter, const tensor<T>& input, const tensor<T>& weights)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc, weights.desc)};
}

struct verify_forward_conv
{
    template<class T>
    std::vector<T> cpu(const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int bias = 0)
    {
        auto out = get_output_tensor(filter, input, weights);

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = mlopen::tie4(input.desc.GetLengths());

        int wei_h, wei_w;
        std::tie(std::ignore, std::ignore, wei_h, wei_w) = mlopen::tie4(weights.desc.GetLengths());

        out.par_for_each([&](int o, int w, int i, int j)
        {
            const int in_off_h = i * filter.v;
            const int in_off_w = j * filter.u;

            T acc = bias;
            ford(in_c, wei_h, wei_w)([&](int k, int x, int y)
            {
                const int in_x = in_off_h - filter.pad_h + x;
                const int in_y = in_off_w - filter.pad_w + y;
                if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w) {
                    acc += input(o, k, in_x, in_y) * weights(w, k, x, y);
                }
            });
            out(o, w, i, j) = acc;
        });
        return out.data;
    }

    template<class T>
    std::vector<T> gpu(const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int /* bias */ = 0)
    {
        mlopen::Handle handle;
        auto out = get_output_tensor(filter, input, weights);

        auto in_dev = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Create<T>(out.data.size());

        int ret_algo_count;
        mlopenConvAlgoPerf_t perf;

        int alpha = 1, beta = 1;

        filter.FindConvFwdAlgorithm(handle,
            input.desc,
            in_dev.get(),
            weights.desc,
            wei_dev.get(),
            out.desc,
            out_dev.get(),
            1,
            &ret_algo_count,
            &perf,
            mlopenConvolutionFastest,
            NULL,
            10,
            0); // MD: Not performing exhaustiveSearch by default for now

        filter.ConvolutionForward(handle,
            &alpha,
            input.desc,
            in_dev.get(),
            weights.desc,
            wei_dev.get(),
            mlopenConvolutionFwdAlgoDirect,
            &beta,
            out.desc,
            out_dev.get(),
            NULL,
            0);

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out.data;
    }

    template<class T>
    void fail(float, const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor&, int /*bias*/ = 0)
    {
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Weights tensor: " << weights.desc.ToString() << std::endl;
    }
    
};

struct verify_conv_filter
{
    template<class T>
    void operator()(const tensor<T>& input, const tensor<T>& weights) const
    {
        mlopen::ConvolutionDescriptor filter{1, 1};
        verify(verify_forward_conv{}, input, weights, filter);
    }
};

int main() {
    // mlopen::Handle handle;
    auto g0 = [](int, int, int, int) { return 0; };
    auto g1 = [](int, int, int, int) { return 1; };
    auto g_id = [](int, int, int h, int w) { return h == w ? 1 : 0; };
    auto g = [](int n, int c, int h, int w)
    {
        double x = (547*n+701*c+877*h+1049*w+173)%1223;
        return x/691.0;
    };
    (void)g0;
    (void)g1;
    (void)g_id;
    (void)g;
#if MLOPEN_TEST_ALL
    printf("verify_all\n");
    generate_all<float, network_visitor>(verify_conv_filter{}, g0,g1, g_id, g);
#else
    printf("verify_one\n");
    generate_one<float>(verify_conv_filter{}, {16, 32, 8, 8}, {64, 32, 5, 5}, g);
#endif
}
