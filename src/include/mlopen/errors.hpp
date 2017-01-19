#ifndef GUARD_MLOPEN_ERRORS_HPP
#define GUARD_MLOPEN_ERRORS_HPP

#include <exception>
#include <string>
#include <iostream>
#include <tuple>
#include <mlopen.h>
#include <mlopen/returns.hpp>

namespace mlopen {

struct Exception : std::exception
{
    std::string message;
    mlopenStatus_t status;
    Exception(const std::string& msg="")
    : message(msg), status(mlopenStatusUnknownError)
    {}

    Exception(mlopenStatus_t s, const std::string& msg="")
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

#define MLOPEN_THROW(...) throw mlopen::Exception(__VA_ARGS__).SetContext(__FILE__, __LINE__)
#define MLOPEN_THROW_CL_STATUS(...) MLOPEN_THROW(mlopenStatusUnknownError, mlopen::OpenCLErrorMessage(__VA_ARGS__))
#define MLOPEN_THROW_HIP_STATUS(...) MLOPEN_THROW(mlopenStatusUnknownError, mlopen::HIPErrorMessage(__VA_ARGS__))

// TODO(paul): Debug builds should leave the exception uncaught
template<class F>
mlopenStatus_t try_(F f)
{
    try 
    {
        f();
    }
    catch(const Exception& ex)
    {
        std::cerr << "MLOpen Error: " << ex.what() << std::endl;
        return ex.status;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "MLOpen Error: " << ex.what() << std::endl;
        return mlopenStatusUnknownError;
    }
    catch(...)
    {
        return mlopenStatusUnknownError;
    }
    return mlopenStatusSuccess;
}

namespace detail {
template<int N>
struct rank : rank<N-1> {};

template<>
struct rank<0> {};    


template<class T>
T& deref_impl(rank<0>, T& x)
{
    return x;
}

template<class T>
auto deref_impl(rank<1>, T& x) -> decltype(mlopen_get_object(x))
{
    return mlopen_get_object(x);
}

}  // namespace detail

template<class T>
auto deref(T& x, mlopenStatus_t err=mlopenStatusBadParm) -> decltype((x == nullptr), detail::deref_impl(detail::rank<1>{}, *x))
{
    if (x == nullptr) { MLOPEN_THROW(err, "Dereferencing nullptr");
}
    return detail::deref_impl(detail::rank<1>{}, *x);
}

template<class... Ts>
auto tie_deref(Ts&... xs) MLOPEN_RETURNS(std::tie(mlopen::deref(xs)...));

} // namespace mlopen

#endif
