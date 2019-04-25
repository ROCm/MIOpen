/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_ERRORS_HPP
#define GUARD_MIOPEN_ERRORS_HPP

#include <exception>
#include <iostream>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/returns.hpp>
#include <string>
#include <tuple>

namespace miopen {

struct Exception : std::exception
{
    std::string message;
    miopenStatus_t status;
    Exception(const std::string& msg = "") : message(msg), status(miopenStatusUnknownError) {}

    Exception(miopenStatus_t s, const std::string& msg = "") : message(msg), status(s) {}

    Exception SetContext(const std::string& file, int line)
    {
        message = file + ":" + std::to_string(line) + ": " + message;
        return *this;
    }

    // Workaround for HIP, this must be inline
    const char* what() const noexcept override { return message.c_str(); }
};

std::string OpenCLErrorMessage(int error, const std::string& msg = "");
std::string HIPErrorMessage(int error, const std::string& msg = "");

#define MIOPEN_THROW(...)                                                    \
    do                                                                       \
    {                                                                        \
        throw miopen::Exception(__VA_ARGS__).SetContext(__FILE__, __LINE__); \
    } while(false)
#define MIOPEN_THROW_CL_STATUS(...) \
    MIOPEN_THROW(miopenStatusUnknownError, miopen::OpenCLErrorMessage(__VA_ARGS__))
#define MIOPEN_THROW_HIP_STATUS(...) \
    MIOPEN_THROW(miopenStatusUnknownError, miopen::HIPErrorMessage(__VA_ARGS__))

// TODO(paul): Debug builds should leave the exception uncaught
template <class F>
miopenStatus_t try_(F f, bool output = true)
{
    try
    {
        f();
    }
    catch(const Exception& ex)
    {
        if(output)
            std::cerr << "MIOpen Error: " << ex.what() << std::endl;
        return ex.status;
    }
    catch(const std::exception& ex)
    {
        if(output)
            std::cerr << "MIOpen Error: " << ex.what() << std::endl;
        return miopenStatusUnknownError;
    }
    catch(...)
    {
        return miopenStatusUnknownError;
    }
    return miopenStatusSuccess;
}

template <class T>
auto deref(T&& x, miopenStatus_t err = miopenStatusBadParm)
    -> decltype((x == nullptr), get_object(*x))
{
    if(x == nullptr)
    {
        MIOPEN_THROW(err, "Dereferencing nullptr");
    }
    return get_object(*x);
}

template <class... Ts>
auto tie_deref(Ts&... xs) MIOPEN_RETURNS(std::tie(miopen::deref(xs)...));

} // namespace miopen

#endif
