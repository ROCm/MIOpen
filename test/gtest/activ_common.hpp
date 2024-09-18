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
namespace activ {

template <class T1,
          class T2,
          std::enable_if_t<(std::is_unsigned_v<T1> && std::is_unsigned_v<T2>), bool> = false>
auto Ceil(T1 val, T2 div)
{
    assert(div != 0);
    return (val + div - 1) / div;
}

namespace activ_func {

/// \note static_cast in below functions is needed to speed up execution time for CPU non-native
/// data types
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
        return std::clamp(static_cast<Tparam>(x), static_cast<Tparam>(0), alpha);
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

enum class Direction
{
    Forward,
    Backward
};

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
void DoCpuActivationForwardPacked(std::size_t offset,
                                  std::size_t end,
                                  Tparam alpha,
                                  Tparam beta,
                                  Tparam gamma,
                                  const tensor<T>& x,
                                  tensor<T>& y)
{
    for(std::size_t i = offset; i < end; i++)
        y.data[i] = A::Forward(alpha, beta, gamma, x.data[i]);
}

template <class A, class Tparam, class T>
void DoCpuActivationBackwardPacked(std::size_t offset,
                                   std::size_t end,
                                   Tparam alpha,
                                   Tparam beta,
                                   Tparam gamma,
                                   const tensor<T>& y,
                                   const tensor<T>& dy,
                                   const tensor<T>& x,
                                   tensor<T>& dx)
{
    for(std::size_t i = offset; i < end; i++)
        dx.data[i] = A::Backward(alpha, beta, gamma, dy.data[i], x.data[i], y.data[i]);
}

template <Direction direction, class A, class... Ts>
void DoCpuActivationPacked(std::size_t offset, std::size_t end, Ts&&... xs)
{
    if constexpr(direction == Direction::Forward)
        DoCpuActivationForwardPacked<A>(offset, end, xs...);
    else
        DoCpuActivationBackwardPacked<A>(offset, end, xs...);
}

template <Direction direction, class A, class... Ts>
void CpuActivationPackedSingleThread(std::size_t num_items, Ts&&... xs)
{
    DoCpuActivationPacked<direction, A>(0, num_items, xs...);
}

template <Direction direction, class A, class... Ts>
void CpuActivationPackedMultiThread(std::size_t num_items, Ts&&... xs)
{
    const std::size_t max_num_items_per_job = 16 * 1024 * 1024;
    const std::size_t num_jobs              = Ceil(num_items, max_num_items_per_job);
    const auto num_threads                  = CpuActivationGetNumThreads(num_jobs);
    if(num_threads == 1)
    {
        CpuActivationPackedSingleThread<direction, A>(num_items, xs...);
        return;
    }

    const std::size_t max_num_jobs_per_thread  = Ceil(num_jobs, num_threads);
    const std::size_t max_num_items_per_thread = max_num_items_per_job * max_num_jobs_per_thread;
    const std::size_t remainder                = num_items % max_num_items_per_thread;
    const auto num_async_threads               = num_threads - 1;

    auto func_async = [&](unsigned thread_num) {
        const auto offset = max_num_items_per_thread * thread_num;
        const auto end    = offset + max_num_items_per_thread;
        DoCpuActivationPacked<direction, A>(offset, end, xs...);
    };

    auto func_remainder = [&]() {
        const auto offset = max_num_items_per_thread * num_async_threads;
        const auto end    = offset + remainder;
        DoCpuActivationPacked<direction, A>(offset, end, xs...);
    };

    std::vector<decltype(std::async(func_async, 0))> threads;
    for(unsigned i = 0; i < num_async_threads; i++)
        threads.push_back(std::async(std::launch::async, func_async, i));

    if(remainder)
        func_remainder();
    else
        func_async(num_async_threads);

    for(auto& thread : threads)
        thread.wait();
}

template <Direction direction, class A, class... Ts>
void CpuActivationPacked(std::size_t num_items, Ts&&... xs)
{
    CpuActivationPackedMultiThread<direction, A>(num_items, xs...);
}

template <Direction direction, class A, class... Ts>
void CpuActivationNonPacked(std::size_t num_items, Ts&&... xs)
{
    throw std::runtime_error("CpuActivationNonPacked is not implemented yet");
}

template <Direction direction, bool is_packed, class A, class... Ts>
void CpuActivation(std::size_t num_items, Ts&&... xs)
{
    if constexpr(is_packed)
        CpuActivationPacked<direction, A>(num_items, xs...);
    else
        CpuActivationNonPacked<direction, A>(num_items, xs...);
}

template <Direction direction, bool is_packed, class... Ts>
void CpuActivation(miopenActivationMode_t m, std::size_t num_items, Ts&&... xs)
{
    switch(m)
    {
    case miopenActivationPASTHRU:
        CpuActivation<direction, is_packed, activ_func::ActivationPASTHRU>(num_items, xs...);
        break;
    case miopenActivationLOGISTIC:
        CpuActivation<direction, is_packed, activ_func::ActivationLOGISTIC>(num_items, xs...);
        break;
    case miopenActivationTANH:
        CpuActivation<direction, is_packed, activ_func::ActivationTANH>(num_items, xs...);
        break;
    case miopenActivationRELU:
        CpuActivation<direction, is_packed, activ_func::ActivationRELU>(num_items, xs...);
        break;
    case miopenActivationSOFTRELU:
        CpuActivation<direction, is_packed, activ_func::ActivationSOFTRELU>(num_items, xs...);
        break;
    case miopenActivationABS:
        CpuActivation<direction, is_packed, activ_func::ActivationABS>(num_items, xs...);
        break;
    case miopenActivationPOWER:
        CpuActivation<direction, is_packed, activ_func::ActivationPOWER>(num_items, xs...);
        break;
    case miopenActivationCLIPPEDRELU:
        CpuActivation<direction, is_packed, activ_func::ActivationCLIPPEDRELU>(num_items, xs...);
        break;
    case miopenActivationLEAKYRELU:
        CpuActivation<direction, is_packed, activ_func::ActivationLEAKYRELU>(num_items, xs...);
        break;
    case miopenActivationELU:
        CpuActivation<direction, is_packed, activ_func::ActivationELU>(num_items, xs...);
        break;
    default: throw std::runtime_error("Unknown activation mode");
    }
}

template <class T>
void CpuActivationForward(miopenActivationMode_t m,
                          double alpha,
                          double beta,
                          double gamma,
                          const tensor<T>& x,
                          tensor<T>& y)
{
    using Tparam = typename ActivationParamDataType<T>::Type;

    if(x.desc.GetElementSize() != y.desc.GetElementSize())
        throw std::runtime_error("x.desc.GetElementSize() != y.desc.GetElementSize()");

    Tparam p_alpha = alpha;
    Tparam p_beta  = beta;
    Tparam p_gamma = gamma;

    if(x.desc.IsPacked() && y.desc.IsPacked())
    {
        const auto num_items = x.data.size();
        CpuActivation<Direction::Forward, true>(m, num_items, p_alpha, p_beta, p_gamma, x, y);
    }
    else
    {
        CpuActivation<Direction::Forward, false>(m, 0, p_alpha, p_beta, p_gamma, x, y);
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
    using Tparam = typename ActivationParamDataType<T>::Type;

    if(x.desc.GetElementSize() != y.desc.GetElementSize())
        throw std::runtime_error("x.desc.GetElementSize() != y.desc.GetElementSize()");
    if(dx.desc.GetElementSize() != dy.desc.GetElementSize())
        throw std::runtime_error("dx.desc.GetElementSize() != dy.desc.GetElementSize()");
    if(x.desc.GetElementSize() != dx.desc.GetElementSize())
        throw std::runtime_error("x.desc.GetElementSize() != dx.desc.GetElementSize()");

    Tparam p_alpha = alpha;
    Tparam p_beta  = beta;
    Tparam p_gamma = gamma;

    if(y.desc.IsPacked() && dy.desc.IsPacked() && x.desc.IsPacked() && dx.desc.IsPacked())
    {
        const auto num_items = dy.data.size();
        CpuActivation<Direction::Backward, true>(
            m, num_items, p_alpha, p_beta, p_gamma, y, dy, x, dx);
    }
    else
    {
        CpuActivation<Direction::Backward, false>(m, 0, p_alpha, p_beta, p_gamma, y, dy, x, dx);
    }
}

} // namespace activ
} // namespace tests
} // namespace miopen
