#include <miopen/kernel_warnings.hpp>
#include <sstream>
#include <iterator>
#include <miopen/config.h>

namespace miopen {

std::vector<std::string> KernelWarnings()
{
    return {
        "-Weverything",
        "-Wno-shorten-64-to-32",
        "-Wno-unused-macros",
        "-Wno-unused-function",
        "-Wno-sign-compare",
        "-Wno-reserved-id-macro",
        "-Wno-sign-conversion",
        "-Wno-missing-prototypes",
        "-Wno-cast-qual",
        "-Wno-cast-align",
        "-Wno-conversion",
        "-Wno-double-promotion",
        "-Wno-float-equal",
    };
}

std::string MakeKernelWarningsString()
{
    std::string result;
#if MIOPEN_BACKEND_OPENCL
    std::string prefix = " -Wf,";
    
#else
    std::string prefix = " ";
#endif
    for(auto&& x:KernelWarnings())
        result += prefix + x;
    return result;
}

const std::string& KernelWarningsString()
{
    static const std::string result = MakeKernelWarningsString();
    return result;
}

} // namespace miopen
