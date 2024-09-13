/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#pragma once

#include <cmath>
#include <future>
#include <numeric>
#include <stdexcept>
#include <thread>

#include <miopen/miopen.h>

#include "../tensor_holder.hpp"

namespace miopen {
namespace tests {

namespace activ_func {

class ActivationPASTHRU
{
public:
    template <class T, class Tparam>
    static T Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        return x;
    }

    template <class T, class Tparam>
    static T Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        return dy;
    }
};

class ActivationLOGISTIC
{
public:
    template <class T, class Tparam>
    static Tparam Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        return 1 / (1 + std::exp(-x));
    }

    template <class T, class Tparam>
    static Tparam Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        return static_cast<Tparam>(dy) * y * (static_cast<Tparam>(1) - y);
    }
};

class ActivationTANH
{
public:
    template <class T, class Tparam>
    static Tparam Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        return beta * std::tanh(alpha * x);
    }

    template <class T, class Tparam>
    static Tparam Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        return dy * alpha * (beta - static_cast<Tparam>(y) * y / beta);
    }
};

class ActivationRELU
{
public:
    template <class T, class Tparam>
    static T Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        return (x > static_cast<Tparam>(0)) ? x : static_cast<T>(0);
    }

    template <class T, class Tparam>
    static T Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        return (x > static_cast<Tparam>(0)) ? dy : static_cast<T>(0);
    }
};

class ActivationSOFTRELU
{
public:
    template <class T, class Tparam>
    static Tparam Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        return std::log1p(std::exp(x));
    }

    template <class T, class Tparam>
    static Tparam Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        const Tparam threshold = 50.0;
        const Tparam expval    = std::exp(std::min(static_cast<Tparam>(x), threshold));
        return dy * expval / (expval + 1);
    }
};

class ActivationABS
{
public:
    template <class T, class Tparam>
    static Tparam Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        return std::abs(x);
    }

    template <class T, class Tparam>
    static Tparam Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        return dy * static_cast<Tparam>((x > static_cast<Tparam>(0)) ? 1 : -1);
    }
};

class ActivationPOWER
{
public:
    template <class T, class Tparam>
    static Tparam Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        const auto v = alpha + beta * x;
        return v <= std::numeric_limits<decltype(v)>::epsilon() ? 0 : std::pow(v, gamma);
    }

    template <class T, class Tparam>
    static Tparam Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        const auto v = alpha + beta * x;
        return v <= std::numeric_limits<decltype(v)>::epsilon() ? 0 : gamma * beta * y / v;
    }
};

class ActivationCLIPPEDRELU
{
public:
    template <class T, class Tparam>
    static Tparam Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        return std::min(alpha, std::max(static_cast<Tparam>(0), static_cast<Tparam>(x)));
    }

    template <class T, class Tparam>
    static T Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        Tparam x_native = x;
        return (x_native > 0 && x_native <= alpha) ? dy : static_cast<T>(0);
    }
};

class ActivationLEAKYRELU
{
public:
    template <class T, class Tparam>
    static Tparam Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        Tparam x_native = x;
        return (x_native > 0) ? x_native : x_native * alpha;
    }

    template <class T, class Tparam>
    static Tparam Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        return dy * ((x > static_cast<Tparam>(0)) ? 1 : alpha);
    }
};

class ActivationELU
{
public:
    template <class T, class Tparam>
    static Tparam Forward(Tparam alpha, Tparam beta, Tparam gamma, T x)
    {
        Tparam x_native = x;
        return (x_native > 0) ? x_native : alpha * std::expm1(x_native);
    }

    template <class T, class Tparam>
    static Tparam Backward(Tparam alpha, Tparam beta, Tparam gamma, T dy, T x, T y)
    {
        return dy * ((x > static_cast<Tparam>(0)) ? 1 : y + alpha);
    }
};

} // namespace activ_func

template <class T>
class ActivationParamDataType;

template <>
class ActivationParamDataType<double>
{
public:
    using Type = double; // "Type" must be a native CPU type!
};

template <>
class ActivationParamDataType<float>
{
public:
    using Type = float; // "Type" must be a native CPU type!
};

template <>
class ActivationParamDataType<half_float::half>
{
public:
    using Type = float; // "Type" must be a native CPU type!
};

template <>
class ActivationParamDataType<bfloat16>
{
public:
    using Type = float; // "Type" must be a native CPU type!
};

unsigned CpuActivationGetNumThreads(std::size_t num_jobs)
{
    const unsigned max_num_hw_threads = std::thread::hardware_concurrency();
    return std::min(num_jobs, static_cast<std::size_t>(max_num_hw_threads));
}

template <class A, class Tparam, class T>
void CpuActivationForwardPackedSingleThread(
    Tparam alpha, Tparam beta, Tparam gamma, const tensor<T>& x, tensor<T>& y)
{
    const auto len = x.data.size();
    for(std::size_t i = 0; i < len; i++)
        y.data[i] = A::Forward(alpha, beta, gamma, x.data[i]);
}

template <class A, class Tparam, class T>
void CpuActivationBackwardPackedSingleThread(Tparam alpha,
                                             Tparam beta,
                                             Tparam gamma,
                                             const tensor<T>& y,
                                             const tensor<T>& dy,
                                             const tensor<T>& x,
                                             tensor<T>& dx)
{
    const auto len = dy.data.size();
    for(std::size_t i = 0; i < len; i++)
        dx.data[i] = A::Backward(alpha, beta, gamma, dy.data[i], x.data[i], y.data[i]);
}

template <class A, class Tparam, class T>
void CpuActivationForwardPackedMultiThread(
    Tparam alpha, Tparam beta, Tparam gamma, const tensor<T>& x, tensor<T>& y)
{
    const auto num_items                    = x.data.size();
    const std::size_t max_num_items_per_job = 16 * 1024 * 1024;
    const std::size_t num_jobs = (num_items + max_num_items_per_job - 1) / max_num_items_per_job;
    const auto num_threads     = CpuActivationGetNumThreads(num_jobs);
    if(num_threads == 1)
    {
        CpuActivationForwardPackedSingleThread<A, Tparam>(alpha, beta, gamma, x, y);
        return;
    }

    const std::size_t max_num_jobs_per_thread  = (num_jobs + num_threads - 1) / num_threads;
    const std::size_t max_num_items_per_thread = max_num_items_per_job * max_num_jobs_per_thread;
    const std::size_t remainder                = num_items % max_num_items_per_thread;
    const auto num_async_threads               = num_threads - 1;

    auto func_async = [&, max_num_items_per_thread, alpha, beta, gamma](unsigned thread_num) {
        const auto offset = max_num_items_per_thread * thread_num;
        const auto end    = offset + max_num_items_per_thread;
        for(std::size_t i = offset; i < end; i++)
            y.data[i] = A::Forward(alpha, beta, gamma, x.data[i]);
    };

    auto func_reminder =
        [&, max_num_items_per_thread, remainder, num_async_threads, alpha, beta, gamma]() {
            const auto offset = max_num_items_per_thread * num_async_threads;
            const auto end    = offset + (remainder ? remainder : max_num_items_per_thread);
            for(std::size_t i = offset; i < end; i++)
                y.data[i] = A::Forward(alpha, beta, gamma, x.data[i]);
        };

    std::vector<decltype(std::async(func_async, 0))> threads;
    for(unsigned i = 0; i < num_async_threads; i++)
        threads.push_back(std::async(std::launch::async, func_async, i));

    func_reminder();

    for(auto& thread : threads)
        thread.wait();
}

template <class A, class Tparam, class T>
void CpuActivationBackwardPackedMultiThread(Tparam alpha,
                                            Tparam beta,
                                            Tparam gamma,
                                            const tensor<T>& y,
                                            const tensor<T>& dy,
                                            const tensor<T>& x,
                                            tensor<T>& dx)
{
    const auto num_items                    = dy.data.size();
    const std::size_t max_num_items_per_job = 16 * 1024 * 1024;
    const std::size_t num_jobs = (num_items + max_num_items_per_job - 1) / max_num_items_per_job;
    const auto num_threads     = CpuActivationGetNumThreads(num_jobs);
    if(num_threads == 1)
    {
        CpuActivationBackwardPackedSingleThread<A, Tparam>(alpha, beta, gamma, y, dy, x, dx);
        return;
    }

    const std::size_t max_num_jobs_per_thread  = (num_jobs + num_threads - 1) / num_threads;
    const std::size_t max_num_items_per_thread = max_num_items_per_job * max_num_jobs_per_thread;
    const std::size_t remainder                = num_items % max_num_items_per_thread;
    const auto num_async_threads               = num_threads - 1;

    auto func_async = [&, max_num_items_per_thread, alpha, beta, gamma](unsigned thread_num) {
        const auto offset = max_num_items_per_thread * thread_num;
        const auto end    = offset + max_num_items_per_thread;
        for(std::size_t i = offset; i < end; i++)
            dx.data[i] = A::Backward(alpha, beta, gamma, dy.data[i], x.data[i], y.data[i]);
    };

    auto func_reminder =
        [&, max_num_items_per_thread, remainder, num_async_threads, alpha, beta, gamma]() {
            const auto offset = max_num_items_per_thread * num_async_threads;
            const auto end    = offset + (remainder ? remainder : max_num_items_per_thread);
            for(std::size_t i = offset; i < end; i++)
                dx.data[i] = A::Backward(alpha, beta, gamma, dy.data[i], x.data[i], y.data[i]);
        };

    std::vector<decltype(std::async(func_async, 0))> threads;
    for(unsigned i = 0; i < num_async_threads; i++)
        threads.push_back(std::async(std::launch::async, func_async, i));

    func_reminder();

    for(auto& thread : threads)
        thread.wait();
}

template <class A, class Tparam, class T>
void CpuActivationForwardPacked(
    Tparam alpha, Tparam beta, Tparam gamma, const tensor<T>& x, tensor<T>& y)
{
    CpuActivationForwardPackedMultiThread<A, Tparam>(alpha, beta, gamma, x, y);
}

template <class A, class Tparam, class T>
void CpuActivationBackwardPacked(Tparam alpha,
                                 Tparam beta,
                                 Tparam gamma,
                                 const tensor<T>& y,
                                 const tensor<T>& dy,
                                 const tensor<T>& x,
                                 tensor<T>& dx)
{
    CpuActivationBackwardPackedMultiThread<A, Tparam>(alpha, beta, gamma, y, dy, x, dx);
}

template <class A, class Tparam, class T>
void CpuActivationForwardNonPacked(
    Tparam alpha, Tparam beta, Tparam gamma, const tensor<T>& x, tensor<T>& y)
{
    throw std::runtime_error("CpuActivationForwardNonPacked is not implemented yet");
}

template <class A, class Tparam, class T>
void CpuActivationBackwardNonPacked(Tparam alpha,
                                    Tparam beta,
                                    Tparam gamma,
                                    const tensor<T>& y,
                                    const tensor<T>& dy,
                                    const tensor<T>& x,
                                    tensor<T>& dx)
{
    throw std::runtime_error("CpuActivationBackwardNonPacked is not implemented yet");
}

template <class A, class T>
void CpuActivationForward(double alpha, double beta, double gamma, const tensor<T>& x, tensor<T>& y)
{
    using Tparam = typename ActivationParamDataType<T>::Type;

    if(x.desc.IsPacked() && y.desc.IsPacked())
        CpuActivationForwardPacked<A, Tparam>(alpha, beta, gamma, x, y);
    else
        CpuActivationForwardNonPacked<A, Tparam>(alpha, beta, gamma, x, y);
}

template <class A, class T>
void CpuActivationBackward(double alpha,
                           double beta,
                           double gamma,
                           const tensor<T>& y,
                           const tensor<T>& dy,
                           const tensor<T>& x,
                           tensor<T>& dx)
{
    using Tparam = typename ActivationParamDataType<T>::Type;

    if(y.desc.IsPacked() && dy.desc.IsPacked() && x.desc.IsPacked() && dx.desc.IsPacked())
        CpuActivationBackwardPacked<A, Tparam>(alpha, beta, gamma, y, dy, x, dx);
    else
        CpuActivationBackwardNonPacked<A, Tparam>(alpha, beta, gamma, y, dy, x, dx);
}

template <class T>
void CpuActivationForward(miopenActivationMode_t m,
                          double alpha,
                          double beta,
                          double gamma,
                          const tensor<T>& x,
                          tensor<T>& y)
{
    if(x.desc.GetElementSize() != y.desc.GetElementSize())
        throw std::runtime_error("x.desc.GetElementSize() != y.desc.GetElementSize()");

    switch(m)
    {
    case miopenActivationPASTHRU:
        CpuActivationForward<activ_func::ActivationPASTHRU>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationLOGISTIC:
        CpuActivationForward<activ_func::ActivationLOGISTIC>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationTANH:
        CpuActivationForward<activ_func::ActivationTANH>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationRELU:
        CpuActivationForward<activ_func::ActivationRELU>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationSOFTRELU:
        CpuActivationForward<activ_func::ActivationSOFTRELU>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationABS:
        CpuActivationForward<activ_func::ActivationABS>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationPOWER:
        CpuActivationForward<activ_func::ActivationPOWER>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationCLIPPEDRELU:
        CpuActivationForward<activ_func::ActivationCLIPPEDRELU>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationLEAKYRELU:
        CpuActivationForward<activ_func::ActivationLEAKYRELU>(alpha, beta, gamma, x, y);
        break;
    case miopenActivationELU:
        CpuActivationForward<activ_func::ActivationELU>(alpha, beta, gamma, x, y);
        break;
    default: throw std::runtime_error("Unknown activation mode");
    }
}

template <class T>
void CpuActivationBackward(miopenActivationMode_t m,
                           double alpha,
                           double beta,
                           double gamma,
                           const tensor<T>& y,
                           const tensor<T>& dy,
                           const tensor<T>& x,
                           tensor<T>& dx)
{
    if(x.desc.GetElementSize() != y.desc.GetElementSize())
        throw std::runtime_error("x.desc.GetElementSize() != y.desc.GetElementSize()");
    if(dx.desc.GetElementSize() != dy.desc.GetElementSize())
        throw std::runtime_error("dx.desc.GetElementSize() != dy.desc.GetElementSize()");
    if(x.desc.GetElementSize() != dx.desc.GetElementSize())
        throw std::runtime_error("x.desc.GetElementSize() != dx.desc.GetElementSize()");

    switch(m)
    {
    case miopenActivationPASTHRU:
        CpuActivationBackward<activ_func::ActivationPASTHRU>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationLOGISTIC:
        CpuActivationBackward<activ_func::ActivationLOGISTIC>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationTANH:
        CpuActivationBackward<activ_func::ActivationTANH>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationRELU:
        CpuActivationBackward<activ_func::ActivationRELU>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationSOFTRELU:
        CpuActivationBackward<activ_func::ActivationSOFTRELU>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationABS:
        CpuActivationBackward<activ_func::ActivationABS>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationPOWER:
        CpuActivationBackward<activ_func::ActivationPOWER>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationCLIPPEDRELU:
        CpuActivationBackward<activ_func::ActivationCLIPPEDRELU>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationLEAKYRELU:
        CpuActivationBackward<activ_func::ActivationLEAKYRELU>(alpha, beta, gamma, y, dy, x, dx);
        break;
    case miopenActivationELU:
        CpuActivationBackward<activ_func::ActivationELU>(alpha, beta, gamma, y, dy, x, dx);
        break;
    default: throw std::runtime_error("Unknown activation mode");
    }
}

} // namespace tests
} // namespace miopen
