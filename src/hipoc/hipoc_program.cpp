#include <mlopen/hipoc_program.hpp>
#include <mlopen/kernel.hpp>
#include <mlopen/errors.hpp>
#include <mlopen/replace.hpp>

#include <sstream>

#include <unistd.h>

namespace mlopen {

std::string quote(std::string s)
{
    return '"' + s + '"';
}

std::string get_thread_id()
{
    std::stringstream ss;
    ss << std::this_thread::get_id();
    return ss.str();
}

struct tmp_file
{
    std::string name;
    tmp_file() = delete;
    tmp_file(const tmp_file&)=delete;
    tmp_file(std::string s)
    : name("/tmp/" + s + "." + get_thread_id() + "." + std::to_string(std::time(nullptr)))
    {
    }

    ~tmp_file()
    {
        std::remove(name.c_str());
    }
};

void execute(std::string exe, std::string args)
{
    // Compensate for team city's broken build
    std::string compiler = std::string(HIP_OC_COMPILER);
    std::string library_path = compiler.substr(0, compiler.rfind('/')) + std::string("/../../lib/x86_64/");
    // std::cout << "LD_LIBRARY_PATH=" << library_path << std::endl;
    setenv("LD_LIBRARY_PATH", library_path.c_str(), 0);

    std::string cmd = exe + " " + args;
    if (std::system(cmd.c_str()) != 0) MLOPEN_THROW("Can't execute " + cmd);
}

void WriteFile(const std::string& content, const std::string& name)
{
    HIPOCProgram::FilePtr f{std::fopen(name.c_str(), "w")};
    if (std::fwrite(content.c_str(), 1, content.size(), f.get()) != content.size()) 
        MLOPEN_THROW("Failed to write to src file");
}

hipModulePtr CreateModule(const std::string& program_name, std::string params)
{   
    tmp_file src_file{program_name};
    std::string src = GetKernelSrc(program_name);

    WriteFile(src, src_file.name);
        
    execute(HIP_OC_COMPILER, 
        "-march=hsail64 -mdevice=Fiji -cl-denorms-are-zero -save-temps=dump -nobin " + 
        params + " " +
        quote(src_file.name));

    std::string obj_file = src_file.name + "_obj";

    execute(HIP_OC_FINALIZER,
        "-target=8:0:3 -hsail dump_0_Fiji.hsail -output=" +
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
