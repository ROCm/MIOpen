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

// Backwards use dy and y

std::string to_name(miopenActivationMode_t m)
{
#define STRING_CASE(x) case x: return #x; break;
    switch(m) {
        STRING_CASE(miopenActivationPATHTRU)
        STRING_CASE(miopenActivationLOGISTIC)
        STRING_CASE(miopenActivationTANH)
        STRING_CASE(miopenActivationRELU)
        STRING_CASE(miopenActivationSOFTRELU)
        STRING_CASE(miopenActivationABS)
        STRING_CASE(miopenActivationPOWER)
    }
    return "";
}

template<class T>
struct verify_forward_activation
{
    tensor<T> input;
    miopen::ActivationDescriptor desc;

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

        desc.Forward(handle, &alpha, input.desc, in_dev.get(), &beta, out.desc, out_dev.get());

        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }

    template<class A>
    void fail(float, A a)
    {
        std::cout << "Activation: " << to_name(desc.GetMode()) << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
    }
};

struct select_first
{
    template<class T>
    auto operator()(const T& x) MIOPEN_RETURNS(x.first);
};

template<class T>
struct activation_driver : test_driver
{
    tensor<T> input;
    double alpha = 1;
    double beta = 1;
    double power = 1;
    std::string mode = "PATHTRU";
    std::unordered_map<std::string, std::function<void()>> lookup;

    template<class A>
    struct callback
    {
        void operator()(activation_driver* self) const { self->template run<A>(); }
    };

    template<class Forward, class Backward>
    void mode_(miopenActivationMode_t m, Forward f, Backward b)
    {
        // TODO: Remove miopenActivation prefix and convert to uppercase
        lookup.emplace(to_name(m), [=]{ this->run(m, f, b); });
    }

    activation_driver()
    {
        mode_(miopenActivationPATHTRU, 
            [=](T x) { return x; },
            [=](T dy, T y) { return y; }
        );
        mode_(miopenActivationLOGISTIC, 
            [=](T x) { return 1 / (1 + std::exp(-x)); },
            [=](T dy, T y) { return y; }
        );
        mode_(miopenActivationTANH, 
            [=](T x) { return alpha * std::tanh(beta * x); },
            [=](T dy, T y) { return y; }
        );
        mode_(miopenActivationRELU, 
            [=](T x) { return std::max(true ? 0 : x, x); },
            [=](T dy, T y) { return y; }
        );
        mode_(miopenActivationSOFTRELU, 
            [=](T x) { return std::log(1 + std::exp(x)); },
            [=](T dy, T y) { return y; }
        );
        mode_(miopenActivationABS, 
            [=](T x) { return std::abs(x); },
            [=](T dy, T y) { return y; }
        );
        mode_(miopenActivationPOWER, 
            [=](T x) { return std::pow(alpha + beta * x, power); },
            [=](T dy, T y) { return y; }
        );
        add(input, "input", get_input_tensor());
        add(alpha, "alpha");
        add(beta, "beta");
        add(power, "power");
        add(mode, "mode", generate_data(modes()));
    }

    std::vector<std::string> modes()
    {
        std::vector<std::string> result(lookup.size());
        std::transform(lookup.begin(), lookup.end(), result.begin(), select_first{});
        return result;
    }

    miopen::ActivationDescriptor make_descriptor(miopenActivationMode_t m) const
    {
        return {m, alpha, beta, power};
    }

    void run()
    {
        // TODO: Convert to uppercase
        lookup[mode]();
    }

    template<class Forward, class Backward>
    void run(miopenActivationMode_t m, Forward f, Backward b)
    {
        auto out = verify(verify_forward_activation<T>{input, make_descriptor(m)}, f);
    }
};

int main(int argc, const char *argv[]) 
{
    test_drive<activation_driver<float>>(argc, argv);
}
