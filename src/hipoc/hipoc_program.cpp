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

struct tmp_dir
{
    std::string name;
    tmp_dir(std::string prefix)
    {
        std::string s = "/tmp/miopen-" + std::move(prefix) + "-XXXXXX";
        std::vector<char> t(s.begin(), s.end());
        t.push_back(0);
        name = mkdtemp(t.data());
    }

    void execute(std::string exe, std::string args)
    {
        std::string cd = "cd " + this->name + "; ";
        std::string cmd = cd + exe + " " + args + " > /dev/null";
        // std::cout << cmd << std::endl;
        if (std::system(cmd.c_str()) != 0) MLOPEN_THROW("Can't execute " + cmd);
    }

    std::string path(std::string f)
    {
        return name + '/' + f;
    }

    ~tmp_dir()
    {
        if(!name.empty())
        {
            std::string cmd = "rm -rf " + name;
            if (std::system(cmd.c_str()) != 0) MLOPEN_THROW("Can't execute " + cmd);
        }
    }
};

void WriteFile(const std::string& content, const std::string& name)
{
    HIPOCProgram::FilePtr f{std::fopen(name.c_str(), "w")};
    if (std::fwrite(content.c_str(), 1, content.size(), f.get()) != content.size()) 
        MLOPEN_THROW("Failed to write to src file");
}

hipModulePtr CreateModule(const std::string& program_name, std::string params)
{   
    tmp_dir dir{program_name};

    std::string src = GetKernelSrc(program_name);

    WriteFile(src, dir.path(program_name));
        
    std::string bin_file = dir.path(program_name) + ".bin";
    std::string hsaco_file = dir.path(program_name) + ".hsaco";
    std::string obj_file = dir.path(program_name) + ".obj";

#if 0
    execute(HIP_OC_COMPILER, 
        "-march=hsail64 -mdevice=Fiji -output=" + bin_file +  
        params + " " +
        src_file.name);

    execute("/usr/bin/objcopy", 
        "-I elf32-little -j .text -O binary -S " + bin_file +
        " " + hsaco_file
    );
#else
    dir.execute(HIP_OC_COMPILER, 
        "-march=hsail64 -mdevice=Fiji -save-temps=dump -nobin " +  
        params + " " +
        program_name);
    dir.execute(HIP_OC_FINALIZER,
        "-target=8:0:3 -hsail dump_0_Fiji.hsail -output=" + hsaco_file
    );
#endif

    hipModule_t raw_m;
    auto status = hipModuleLoad(&raw_m, hsaco_file.c_str());
    hipModulePtr m{raw_m};
    if (status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Failed creating module");
    return m;
}

HIPOCProgram::HIPOCProgram()
{}
HIPOCProgram::HIPOCProgram(const std::string &program_name, std::string params)
{
    this->module = CreateModule(program_name, params);
}
}
