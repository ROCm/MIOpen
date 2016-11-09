#include <mlopen.h>
#include "test.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <mlopen/tensor.hpp>
#include <mlopen/pooling.hpp>
#include <limits>

// #include "network_data.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "driver.hpp"

template<class T>
tensor<T> get_output_tensor(const mlopen::PoolingDescriptor& filter, const tensor<T>& input)
{
    return tensor<T>{filter.GetForwardOutputTensor(input.desc)};
}

struct verify_max_pooling
{
    template<class T>
    tensor<T> cpu(const tensor<T>& input, const mlopen::PoolingDescriptor& filter)
    {
        auto out = get_output_tensor(filter, input);

        int in_h, in_w;
        std::tie(std::ignore, std::ignore, in_h, in_w) = mlopen::tie4(input.desc.GetLengths());

        int u, v, pad_h, pad_w, window_h, window_w;
        std::tie(u, v) = mlopen::tie2(filter.GetStrides());
        std::tie(pad_h, pad_w) = mlopen::tie2(filter.GetPads());
        std::tie(window_h, window_w) = mlopen::tie2(filter.GetLengths());

        out.par_for_each([&](int o, int w, int i, int j)
        {
            const int in_off_h = i * v;
            const int in_off_w = j * u;

            T acc = std::numeric_limits<T>::lowest();
            ford(window_h, window_w)([&](int x, int y)
            {
                const int in_x = in_off_h - pad_h + x;
                const int in_y = in_off_w - pad_w + y;
                if(in_x >= 0 && in_x < in_h && in_y >= 0 && in_y < in_w) {
                    acc = std::max(acc, input(o, w, in_x, in_y));
                }
            });
            out(o, w, i, j) = acc;
        });
        return out;
    }

    template<class T>
    tensor<T> gpu(const tensor<T>& input, const mlopen::PoolingDescriptor& filter)
    {
        mlopen::Handle handle;
        auto out = get_output_tensor(filter, input);

        auto in_dev = handle.Write(input.data);
        auto out_dev = handle.Create<T>(out.data.size());

        int alpha = 1, beta = 1;
        filter.Forward(
            handle,
            &alpha,
            input.desc,
            in_dev.get(),
            &beta,
            out.desc,
            out_dev.get(),
            false,
            nullptr,
            0
        );

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template<class T>
    void fail(float, const tensor<T>& input, const mlopen::PoolingDescriptor& filter)
    {
        std::cout << "Forward max pooling: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        std::cout << "Output tensor: " << filter.GetForwardOutputTensor(input.desc).ToString() << std::endl;
    }
    
};


struct verify_pooling
{
    template<class T>
    void operator()(const tensor<T>& input) const
    {
        mlopen::PoolingDescriptor filter1{mlopenPoolingMax, {2, 2}, {1, 1}, {0, 0}};
        verify(verify_max_pooling{}, input, filter1);
        
        mlopen::PoolingDescriptor filter2{mlopenPoolingMax, {2, 2}, {1, 1}, {1, 1}};
        verify(verify_max_pooling{}, input, filter2);
    }
};

int main(int argc, const char *argv[]) 
{
    activation_test_drive<verify_pooling>(argc, argv);
}
