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

// #include "network_data.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "driver.hpp"
#include "get_handle.hpp"

template<class T>
tensor<T> get_output_tensor(const mlopen::ConvolutionDescriptor& filter, const tensor<T>& input, const tensor<T>& weights)
{
    assert(filter.GetBackwardOutputTensor(filter.GetForwardOutputTensor(input.desc, weights.desc), weights.desc) == input.desc);
    return tensor<T>{filter.GetForwardOutputTensor(input.desc, weights.desc)};
}

struct verify_forward_conv
{
    template<class T>
    tensor<T> cpu(const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int bias = 0)
    {
        auto out = get_output_tensor(filter, input, weights);

        int in_h, in_w;
        std::tie(std::ignore, std::ignore, in_h, in_w) = mlopen::tie4(input.desc.GetLengths());

        int wei_c, wei_h, wei_w;
        std::tie(std::ignore, wei_c, wei_h, wei_w) = mlopen::tie4(weights.desc.GetLengths());

        out.par_for_each([&](int o, int w, int i, int j)
        {
            const int start_x = i * filter.v - filter.pad_h;
            const int start_y = j * filter.u - filter.pad_w;

            double acc = bias;
            ford(wei_c, wei_h, wei_w)([&](int k, int x, int y)
            {
                const int in_x = start_x + x;
                const int in_y = start_y + y;
                if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w) {
                    acc += input(o, k, in_x, in_y) * weights(w, k, x, y);
                }
            });
            out(o, w, i, j) = acc;
        });
        return out;
    }

    template<class T>
    tensor<T> gpu(const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int /* bias */ = 0)
    {
        auto&& handle = get_handle();
        auto out = get_output_tensor(filter, input, weights);

        auto in_dev = handle.Write(input.data);
        auto wei_dev = handle.Write(weights.data);
        auto out_dev = handle.Create<T>(out.data.size());

		size_t workspace_size = filter.ForwardGetWorkSpaceSize(weights.desc, out.desc);

		std::vector<char> workspace(workspace_size);
		auto workspace_dev = handle.Write(workspace);

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
            workspace_dev.get(),
            workspace_size,
            0); // MD: Not performing exhaustiveSearch by default for now

        filter.ConvolutionForward(handle,
            &alpha,
            input.desc,
            in_dev.get(),
            weights.desc,
            wei_dev.get(),
            perf.fwd_algo,
            &beta,
            out.desc,
            out_dev.get(),
            workspace_dev.get(),
            workspace_size);

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template<class T>
    void fail(float, const tensor<T>& input, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int /*bias*/ = 0)
    {
        std::cout << "Forward convolution: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Output tensor: " << filter.GetForwardOutputTensor(input.desc, weights.desc).ToString() << std::endl;
        std::cout << "Weights tensor: " << weights.desc.ToString() << std::endl;
    }
    
};

template<class T>
tensor<T> get_input_tensor(const mlopen::ConvolutionDescriptor& filter, const tensor<T>& out, const tensor<T>& weights)
{
    assert(filter.GetForwardOutputTensor(filter.GetBackwardOutputTensor(out.desc, weights.desc), weights.desc) == out.desc);
    return tensor<T>{filter.GetBackwardOutputTensor(out.desc, weights.desc)};
}

struct verify_backward_conv
{
    template<class T>
    tensor<T> cpu(const tensor<T>& out, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int /* bias */ = 0)
    {
        auto input = get_input_tensor(filter, out, weights);
        std::fill(input.begin(), input.end(), 0);

        int in_h, in_w;
        std::tie(std::ignore, std::ignore, in_h, in_w) = mlopen::tie4(input.desc.GetLengths());

        int wei_c, wei_h, wei_w;
        std::tie(std::ignore, wei_c, wei_h, wei_w) = mlopen::tie4(weights.desc.GetLengths());

        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) = mlopen::tie4(out.desc.GetLengths());

        par_ford(out_n, wei_c)([&](int o, int k)
        {
            ford(out_c, out_h, out_w, wei_h, wei_w)([&](int w, int i, int j, int x, int y)
            {
                const int start_x = i * filter.v - filter.pad_h;
                const int start_y = j * filter.u - filter.pad_w;
                const int in_x = start_x + x;
                const int in_y = start_y + y;
                if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w) {
                    input(o, k, in_x, in_y) += out(o, w, i, j) * weights(w, k, x, y);
                }
            });
        });
        return input;
    }

    template<class T>
    tensor<T> gpu(const tensor<T>& out, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int /* bias */ = 0)
    {
        auto&& handle = get_handle();
        auto input = get_input_tensor(filter, out, weights);
        std::fill(input.begin(), input.end(), 0);

        auto out_dev = handle.Write(out.data);
        auto wei_dev = handle.Write(weights.data);
        auto in_dev = handle.Create<T>(input.data.size());

        int ret_algo_count;
        mlopenConvAlgoPerf_t perf;

        int alpha = 1, beta = 1;

        filter.FindConvBwdDataAlgorithm(handle,
            out.desc,
            out_dev.get(),
            weights.desc,
            wei_dev.get(),
            input.desc,
            in_dev.get(),
            1,
            &ret_algo_count,
            &perf,
            mlopenConvolutionFastest,
            nullptr,
            10,
            0); // MD: Not performing exhaustiveSearch by default for now

        filter.ConvolutionBackwardData(handle,
            &alpha,
            out.desc,
            out_dev.get(),
            weights.desc,
            wei_dev.get(),
            perf.bwd_data_algo,
            &beta,
            input.desc,
            in_dev.get(),
            nullptr,
            0);

        input.data = handle.Read<T>(in_dev, input.data.size());
        return input;
    }

    template<class T>
    void fail(float, const tensor<T>& output, const tensor<T>& weights, const mlopen::ConvolutionDescriptor& filter, int /*bias*/ = 0)
    {
        std::cout << "Backward convolution: " << std::endl;
        std::cout << "Input tensor: " << filter.GetBackwardOutputTensor(output.desc, weights.desc).ToString() << std::endl;
        std::cout << "Output tensor: " << output.desc.ToString() << std::endl;
        std::cout << "Weights tensor: " << weights.desc.ToString() << std::endl;
    }
    
};

template<class T>
tensor<T> get_weight_tensor(const mlopen::ConvolutionDescriptor& filter, const tensor<T>& input, const tensor<T>& out)
{
    return tensor<T>{filter.GetBackwardWeightsTensor(input.desc, out.desc)};
}

struct verify_backward_weights_conv
{
    template<class T>
    tensor<T> cpu(const tensor<T>& input, const tensor<T>& out, const mlopen::ConvolutionDescriptor& filter, int /* bias */ = 0)
    {
        auto weights = get_weight_tensor(filter, input, out);
        std::fill(weights.begin(), weights.end(), 0);

        int in_h, in_w;
        std::tie(std::ignore, std::ignore, in_h, in_w) = mlopen::tie4(input.desc.GetLengths());

        int wei_c, wei_h, wei_w;
        std::tie(std::ignore, wei_c, wei_h, wei_w) = mlopen::tie4(weights.desc.GetLengths());

        int out_n, out_c, out_h, out_w;
        std::tie(out_n, out_c, out_h, out_w) = mlopen::tie4(out.desc.GetLengths());

        par_ford(out_c, wei_c, wei_h, wei_w)([&](int w, int k, int x, int y)
        {
            double acc = 0.0;
            ford(out_n, out_h, out_w)([&](int o, int i, int j)
            {
                const int start_x = i * filter.v - filter.pad_h;
                const int start_y = j * filter.u - filter.pad_w;
                const int in_x = start_x + x;
                const int in_y = start_y + y;
                if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w) {
                    acc += input(o, k, in_x, in_y) * out(o, w, i, j);
                }
            });
            weights(w, k, x, y) = acc;
        });
        return weights;
    }

    template<class T>
    tensor<T> gpu(const tensor<T>& input, const tensor<T>& out, const mlopen::ConvolutionDescriptor& filter, int /* bias */ = 0)
    {
        auto&& handle = get_handle();
        auto weights = get_weight_tensor(filter, input, out);
        std::fill(weights.begin(), weights.end(), 0);

        auto out_dev = handle.Write(out.data);
        auto wei_dev = handle.Create<T>(weights.data.size());
        auto in_dev = handle.Write(input.data);

        std::size_t workspace_size = filter.ConvolutionBackwardWeightsGetWorkSpaceSize(out.desc, input.desc, weights.desc);

        std::vector<char> workspace(workspace_size);
        auto workspace_dev = handle.Write(workspace);

#if 0
#if MLOPEN_USE_TINYGEMM
        auto algo = mlopenConvolutionBwdWeightsAlgoGEMM;
#else
        auto algo = mlopenConvolutionBwdWeightsAlgoDirect;
#endif
#endif
        int ret_algo_count;
        mlopenConvAlgoPerf_t perf;

        int alpha = 1, beta = 1;
        filter.FindConvBwdWeightsAlgorithm(handle,
            out.desc,
            out_dev.get(),
            input.desc,
            in_dev.get(),
            weights.desc,
            wei_dev.get(),
            1,
            &ret_algo_count,
            &perf,
            mlopenConvolutionFastest,
            workspace_dev.get(),
            workspace_size,
            0); // MD: Not performing exhaustiveSearch by default for now

        filter.ConvolutionBackwardWeights(handle,
            &alpha,
            out.desc,
            out_dev.get(),
            input.desc,
            in_dev.get(),
            perf.bwd_weights_algo,
            &beta,
            weights.desc,
            wei_dev.get(),
            workspace_dev.get(),
            workspace_size);

        weights.data = handle.Read<T>(wei_dev, weights.data.size());
        return weights;
    }

    template<class T>
    void fail(float, const tensor<T>& input, const tensor<T>& output, const mlopen::ConvolutionDescriptor& filter, int /*bias*/ = 0)
    {
        std::cout << "Backward weights convolution: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Output tensor: " << output.desc.ToString() << std::endl;
        std::cout << "Weights tensor: " << filter.GetBackwardWeightsTensor(input.desc, output.desc).ToString() << std::endl;
    }
    
};

template<class T>
struct conv_driver : test_driver
{
    tensor<T> input;
    tensor<T> weights;
    mlopen::ConvolutionDescriptor filter;

    conv_driver()
    {
        add(input, "input", get_input_tensor());
        add(weights, "weights", get_weights_tensor());
        add(filter, "filter", generate_single(mlopen::ConvolutionDescriptor{0, 0}));
    }

    void run()
    {
        if (input.desc.GetLengths().at(1) == weights.desc.GetLengths().at(1))
        {
            auto out_p = verify(verify_forward_conv{}, input, weights, filter);
            for(auto& x:out_p.first) x = (long(x+1)*2) % 17; // Clamp big numbers
            verify(verify_backward_conv{}, out_p.first, weights, filter);
#if MLOPEN_USE_TINYGEMM
            verify(verify_backward_weights_conv{}, input, out_p.first, filter);
#endif
        }
    }
};

int main(int argc, const char *argv[]) 
{
    test_drive<conv_driver<float>>(argc, argv);
}
