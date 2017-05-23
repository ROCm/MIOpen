#include <miopen/miopen.h>
#include "test.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <utility>
#include <iostream>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>
#include <limits>

#include "tensor_holder.hpp"
#include "verify.hpp"
#include "driver.hpp"
#include "get_handle.hpp"

// typedef enum {
//     miopenActivationPATHTRU     = 0,
//     miopenActivationLOGISTIC    = 1, // 1 / (1 + e^-x)
//     miopenActivationTANH        = 2, // a * tanh( b * x)
//     miopenActivationRELU        = 3, // max(0, x)
//     miopenActivationSOFTRELU    = 4, // log(1 + e^x)
//     miopenActivationABS         = 5, // abs(x)
//     miopenActivationPOWER       = 6, // (a + b * x ) ^power
// } miopenActivationMode_t;

template<class A>
struct activation_base
{
    double alpha, beta, power;
    activation_base(double a, double b, double p)
    : alpha(p), beta(b), power(p)
    {}
    miopen::ActivationDescriptor descriptor() const
    {
        return {A::mode(), alpha, beta, power};
    }
};

#define DECLARE_ACTIVATION(name, value, ...) \
struct activation ## name ## _policy \
{ \
    \
};

#define VISIT_ACTIVATIONS(m) \
m(PATHTRU, 0, x) \
m(LOGISTIC, 1, 1 / (1 + std::exp(-x))) \
m(TANH, 2, alpha * std::tanh(beta * x)) \
m(RELU, 3, std::max(0, x)) \
m(SOFTRELU, 4, std::log(1 + e^x)) \
m(ABS, 5, abs(x)) \
m(POWER, 6, (alpha + beta * x ) ^power)

template<class T, class F>
struct verify_forward_activation
{
    ActivationDescriptor filter;
    tensor<T> input;

    template<class T>
    tensor<T> cpu()
    {
        auto out = input;

        input.par_for_each([&](int o, int w, int i, int j) 
        {
            out(o, w, i, j) = f(input(o, w, i, j));
        });

        return out;
    }

    template<class T>
    tensor<T> gpu()
    {
        auto&& handle = get_handle();
        auto out = input;

        // auto out_dev = handle.Write(out.data);

        // int alpha = 1, beta = 1;

        // out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template<class T>
    void fail(float, const tensor<T>& input)
    {
        std::cout << "Acitvation: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

