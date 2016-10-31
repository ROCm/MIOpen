
#include <mlopen.h>
#include "test.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <mlopen/tensor.hpp>
#include <mlopen/convolution.hpp>
#include <mlopen/softmax.hpp>
#include <limits>

#include "tensor_holder.hpp"
#include "verify.hpp"
#include "driver.hpp"


struct verify_forward_sofmax
{
    template<class T>
    tensor<T> cpu(const tensor<T>& input)
    {
        auto out = input;
        std::fill(out.begin(), out.end(), 0);

        int in_n, in_c, in_h, in_w;
        std::tie(in_n, in_c, in_h, in_w) = mlopen::tie4(input.desc.GetLengths());

        par_ford(in_n, in_h, in_w)([&](int o, int i, int j)
        {
            T max_c = -std::numeric_limits<T>::max();
            ford(in_c)([&](int w)
            {
                max_c = std::max(max_c, input(o, w, i, j));
            });

            T sum = 0;
            ford(in_c)([&](int w)
            {
                sum += exp(input(o, w, i, j) - max_c);
            });

            ford(in_c)([&](int w)
            {
                out(o, w, i, j) = exp(input(o, w, i, j) - max_c) / sum;
            });

        });
        return out;
    }

    template<class T>
    tensor<T> gpu(const tensor<T>& input)
    {
        mlopen::Handle handle;
        auto out = input;

        auto out_dev = handle.Write(out.data);

        int alpha = 1, beta = 1;

        SoftmaxForward(handle, &alpha, &beta, input.desc, out_dev.get());

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template<class T>
    void fail(float, const tensor<T>& input)
    {
        std::cout << "Sofmax: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
    
};

struct verify_softmax
{
    template<class T>
    void operator()(const tensor<T>& input) const
    {
        verify(verify_forward_sofmax{}, input);
    }
};

int main(int argc, const char *argv[]) 
{
    activation_test_drive<verify_softmax>(argc, argv);
}

