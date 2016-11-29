#include <mlopen/hipoc_program.hpp>
#include <mlopen/kernel.hpp>
#include <mlopen/errors.hpp>

#include <unistd.h>

namespace mlopen {

void execute(std::string s)
{
    std::cout << s << std::endl;
    std::system(s.c_str());
}

std::string quote(std::string s)
{
    return '"' + s + '"';
}

std::string GetFileName(const HIPOCProgram::FilePtr& f)
{
    auto fd = fileno(f.get());
    std::string path = "/proc/self/fd/" + std::to_string(fd);
    std::array<char, 256> buffer{};
    if (readlink(path.c_str(), buffer.data(), buffer.size()-1) == -1) 
        MLOPEN_THROW("Error reading filename");
    return buffer.data();
}

hipModulePtr CreateModule(const std::string& program_name, std::string params)
{   
    HIPOCProgram::FilePtr tmpsrc{std::tmpfile()};
    std::string src = GetKernelSrc(program_name);
    std::string src_name = GetFileName(tmpsrc);
    
    if (std::fwrite(src.c_str(), 1, src.size(), tmpsrc.get()) != src.size()) 
        MLOPEN_THROW("Failed to write to src file");
    
    execute(HIP_OC_COMPILER + 
        std::string(" -march=hsail64 -mdevice=Fiji -cl-denorms-are-zero -save-temps=dump -nobin ") + 
        params + " " +
        quote(src_name));

    std::string obj_file = src_name + "_obj";

    execute(HIP_OC_FINALIZER +
        std::string(" -target=8:0:3 -hsail dump_0_Fiji.hsail -output=") +
        quote(obj_file)
    );

    hipModule_t raw_m;
    auto status = hipModuleLoad(&raw_m, obj_file.c_str());
    hipModulePtr m{raw_m};
    std::remove(obj_file.c_str());
    if (status != hipSuccess) MLOPEN_THROW("Failed creating module");
    return m;
}

HIPOCProgram::HIPOCProgram()
{}
HIPOCProgram::HIPOCProgram(const std::string &program_name, std::string params)
{
    this->module = CreateModule(program_name, params);
}
}
