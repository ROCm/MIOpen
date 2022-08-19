#include "fin.hpp"

namespace fin {

template <>
miopenDataType_t GetDataType<int8_t>()
{
    return miopenInt8;
}
template <>
miopenDataType_t GetDataType<float>()
{
    return miopenFloat;
}
template <>
miopenDataType_t GetDataType<float16>()
{
    return miopenHalf;
}
template <>
miopenDataType_t GetDataType<bfloat16>()
{
    return miopenBFloat16;
}
[[gnu::noreturn]] void BaseFin::Usage()
{
    std::cout << "Usage: ./MIOpenFin *base_arg* *other_args*\n";
    std::cout << "Supported Base Arguments: conv[fp16][bfp16]\n";
    exit(0);
}
void PadBufferSize(size_t& sz, int datatype_sz)
{
    size_t page_sz = (2 * 1024 * 1024) / datatype_sz;
    if(sz % page_sz != 0)
    {
        sz = ((sz + page_sz) / page_sz) * page_sz;
    }
}

std::string BaseFin::ParseBaseArg(const int argc, const char* argv[])
{
    if(argc < 2)
    {
        std::cout << "Invalid Number of Input Arguments\n";
        Usage();
    }

    std::string arg = argv[1];

    if(arg != "conv" && arg != "convfp16" && arg != "convbfp16")
    {
        std::cout << "Invalid Base Input Argument\n";
        Usage();
    }
    else if(arg == "-h" || arg == "--help" || arg == "-?")
        Usage();
    else
        return arg;
}
void BaseFin::InitNoGpuHandle(miopen::Handle& handle,
                              const std::string& arch,
                              const unsigned long num_cu)
{
#if MIOPEN_MODE_NOGPU
    handle.impl->device_name        = arch;
    handle.impl->num_cu             = num_cu;
    handle.impl->max_mem_alloc_size = 32UL * 1024 * 1024 * 1024; // 32 GB
    handle.impl->global_mem_size    = 32UL * 1024 * 1024 * 1024;
    handle.impl->target_properties.Init(&handle);
#else
    std::ignore = handle;
    std::ignore = arch;
    std::ignore = num_cu;
#endif
}

void BaseFin::VerifyDevProps(const std::string& in_arch, const unsigned long in_num_cu)
{
    std::cerr << "Verifying device properties" << std::endl;
    std::string arch    = in_arch;
    arch                = arch.substr(0, arch.find(':'));
    const size_t num_cu = in_num_cu;
    std::ignore         = num_cu;
    if(arch == "gfx900")
    {
        assert(num_cu == 56 || num_cu == 64);
    }
    else if(arch == "gfx906")
    {
        assert(num_cu == 60 || num_cu == 64);
    }
    else if(arch == "gfx908")
    {
        assert(num_cu == 120);
    }
    else if(arch == "gfx1030")
    {
        assert(num_cu == 72 || num_cu == 36);
    }
    else if(arch == "gfx90a")
    {
        assert(num_cu == 110 || num_cu == 104);
    }
    else
        throw std::runtime_error("Invalid Arch Name");
}
} // namespace fin
