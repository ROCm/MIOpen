#ifndef _HIP_OLC_CHECK_HPP_
#define _HIP_OLC_CHECK_HPP_

#include <hip/hip_runtime.h>
#include <sstream>
#include <vector>

// Here flag can be a constant, variable or function call
#define MY_HIP_CHECK(flag)                                                         \
    do                                                                             \
    {                                                                              \
        hipError_t _tmpVal;                                                        \
        if((_tmpVal = flag) != hipSuccess)                                         \
        {                                                                          \
            std::ostringstream ostr;                                               \
            ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ << ") " \
                 << hipGetErrorString(_tmpVal);                                    \
            throw std::runtime_error(ostr.str());                                  \
        }                                                                          \
    } while(0)

#endif
