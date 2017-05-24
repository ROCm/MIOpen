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

#define VISIT_ACTIVATIONS(m) \
m(PATHTRU, x) \
m(LOGISTIC, 1 / (1 + std::exp(-x))) \
m(TANH, alpha * std::tanh(beta * x)) \
m(RELU, std::max(true ? 0 : x, x)) \
m(SOFTRELU, std::log(1 + std::exp(x))) \
m(ABS, std::abs(x)) \
m(POWER, std::pow(alpha + beta * x, power))

template<class A>
struct activation
{
    double alpha, beta, power;
    activation(double a, double b, double p)
    : alpha(a), beta(b), power(p)
    {}
    miopen::ActivationDescriptor descriptor() const
    {
        return {A::mode(), alpha, beta, power};
    }

    template<class T>
    T operator()(T x) const
    {
        return A::apply(alpha, beta, power, x);
    }

    const char * name() const
    {
        return A::str();
    }
};

#define DECLARE_ACTIVATION_POLICY(name, ...) \
struct activation ## name ## _policy \
{ \
   static miopenActivationMode_t mode() { return miopenActivation##name; } \
   static const char * str() { return #name; } \
   template<class X, class T> \
   static T apply(X alpha, X beta, X power, T x) { (void)alpha;(void)beta;(void)power;return __VA_ARGS__; } \
};
VISIT_ACTIVATIONS(DECLARE_ACTIVATION_POLICY)



template<class T>
struct verify_forward_activation
{
    tensor<T> input;

    template<class A>
    tensor<T> cpu(A a)
    {
        auto out = input;

        input.par_for_each([&](int o, int w, int i, int j) 
        {
            out(o, w, i, j) = a(input(o, w, i, j));
        });

        return out;
    }

    template<class A>
    tensor<T> gpu(A a)
    {
        auto&& handle = get_handle();
        auto out = input;
        auto in_dev = handle.Write(input.data);
        auto out_dev = handle.Write(out.data);

        int alpha = 1, beta = 1;

        auto desc = a.descriptor();
        desc.Forward(handle, &alpha, input.desc, in_dev.get(), &beta, out.desc, out_dev.get());

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template<class A>
    void fail(float, A a)
    {
        std::cout << "Activation: " << a.name() << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

#define LOOKUP_ACTIVATION(name, ...) {#name, callback<activation ## name ## _policy>{} },
template<class T>
struct activation_driver : test_driver
{
    tensor<T> input;
    double alpha = 1;
    double beta = 1;
    double power = 1;
    std::string mode = "PATHTRU";
    std::unordered_map<std::string, std::function<void(activation_driver*)>> lookup;

    template<class A>
    struct callback
    {
        void operator()(activation_driver* self) const { self->template run<A>(); }
    };

    activation_driver()
    {
        add(input, "input", get_input_tensor());
        add(alpha, "alpha");
        add(beta, "beta");
        add(power, "power");
        add(mode, "mode");
        lookup = { VISIT_ACTIVATIONS(LOOKUP_ACTIVATION) };
    }

    void run()
    {
        lookup[mode](this);
    }

    template<class A>
    void run()
    {
        auto out = verify(verify_forward_activation<T>{input}, activation<A>{alpha, beta, power});
    }
};

int main(int argc, const char *argv[]) 
{
    test_drive<activation_driver<float>>(argc, argv);
}
