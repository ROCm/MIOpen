#ifndef GUARD_MLOPEN_ERRORS_HPP
#define GUARD_MLOPEN_ERRORS_HPP

#include <exception>
#include <string>
#include <iostream>
#include <MLOpen.h>

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


    Exception& SetContext(const std::string& file, int line)
    {
        message = file + ":" + std::to_string(line) + ": " + message;
        return *this;
    }

    virtual const char* what() const noexcept;

};

#define MLOPEN_THROW(...) throw mlopen::Exception(__VA_ARGS__).SetContext(__FILE__, __LINE__)

// TODO: Debug builds should leave the exception uncaught
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

template<class T>
auto deref(T&& x, mlopenStatus_t err=mlopenStatusBadParm) -> decltype(*x)
{
    if (x == nullptr) MLOPEN_THROW(err, "Dereferencing nullptr");
    return *x;
}

}

#endif