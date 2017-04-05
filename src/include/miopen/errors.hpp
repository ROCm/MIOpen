#ifndef GUARD_MIOPEN_ERRORS_HPP
#define GUARD_MIOPEN_ERRORS_HPP

#include <exception>
#include <string>
#include <iostream>
#include <tuple>
#include <miopen.h>
#include <miopen/object.hpp>
#include <miopen/returns.hpp>

namespace miopen {

struct Exception : std::exception
{
    std::string message;
    miopenStatus_t status;
    Exception(const std::string& msg="")
    : message(msg), status(miopenStatusUnknownError)
    {}

    Exception(miopenStatus_t s, const std::string& msg="")
    : message(msg), status(s)
    {}


    Exception SetContext(const std::string& file, int line)
    {
        message = file + ":" + std::to_string(line) + ": " + message;
        return *this;
    }

    const char* what() const noexcept override;

};

std::string OpenCLErrorMessage(int error, const std::string& msg="");
std::string HIPErrorMessage(int error, const std::string& msg="");

#define MIOPEN_THROW(...) throw miopen::Exception(__VA_ARGS__).SetContext(__FILE__, __LINE__)
#define MIOPEN_THROW_CL_STATUS(...) MIOPEN_THROW(miopenStatusUnknownError, miopen::OpenCLErrorMessage(__VA_ARGS__))
#define MIOPEN_THROW_HIP_STATUS(...) MIOPEN_THROW(miopenStatusUnknownError, miopen::HIPErrorMessage(__VA_ARGS__))

// TODO(paul): Debug builds should leave the exception uncaught
template<class F>
miopenStatus_t try_(F f)
{
    try 
    {
        f();
    }
    catch(const Exception& ex)
    {
        std::cerr << "MIOpen Error: " << ex.what() << std::endl;
        return ex.status;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "MIOpen Error: " << ex.what() << std::endl;
        return miopenStatusUnknownError;
    }
    catch(...)
    {
        return miopenStatusUnknownError;
    }
    return miopenStatusSuccess;
}


template<class T>
auto deref(T& x, miopenStatus_t err=miopenStatusBadParm) -> decltype((x == nullptr), get_object(*x))
{
    if (x == nullptr) { MIOPEN_THROW(err, "Dereferencing nullptr"); }
    return get_object(*x);
}

template<class... Ts>
auto tie_deref(Ts&... xs) MIOPEN_RETURNS(std::tie(miopen::deref(xs)...));

} // namespace miopen

#endif
