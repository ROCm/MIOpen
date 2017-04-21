#include <miopen/hipoc_program.hpp>
#include <miopen/kernel.hpp>
#include <miopen/errors.hpp>
#include <miopen/stringutils.hpp>

#include <sstream>

#include <unistd.h>

namespace miopen {

std::string quote(std::string s)
{
    return '"' + s + '"';
}

void system_cmd(std::string cmd)
{
    // std::cout << cmd << std::endl;
    // We shouldn't call system commands
#ifdef MIOPEN_USE_CLANG_TIDY
    (void)cmd;
#else
    if (std::system(cmd.c_str()) != 0) MIOPEN_THROW("Can't execute " + cmd);
#endif
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
        std::string cmd = cd + exe + " " + args;// + " > /dev/null";
        system_cmd(cmd);
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
            system_cmd(cmd);
        }
    }
};

void WriteFile(const std::string& content, const std::string& name)
{
    // std::cerr << "Write file: " << name << std::endl;
    HIPOCProgram::FilePtr f{std::fopen(name.c_str(), "w")};
    if (std::fwrite(content.c_str(), 1, content.size(), f.get()) != content.size()) 
        MIOPEN_THROW("Failed to write to src file");
}

hipModulePtr CreateModule(const std::string& program_name, std::string params, bool is_kernel_str)
{
    std::string filename = is_kernel_str ? "tinygemm.cl" : program_name;
    tmp_dir dir{filename};
    std::string bin_file = dir.path(filename) + ".bin";
    std::string hsaco_file = dir.path(filename) + ".o";
    std::string obj_file = dir.path(filename) + ".obj";

    std::string src = is_kernel_str ? program_name : GetKernelSrc(program_name);
    if (!is_kernel_str && miopen::EndsWith(program_name, ".so"))
    {
        WriteFile(src, hsaco_file);        
    }
    else
    {

        WriteFile(src, dir.path(filename));

#if MIOPEN_BUILD_DEV
        params += " -Werror -Weverything -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-unused-function -Wno-sign-compare -Wno-reserved-id-macro -Wno-sign-conversion -Wno-missing-prototypes -Wno-cast-qual -Wno-cast-align -Wno-conversion -Wno-double-promotion";
#else
        params += " -Wno-everything";
#endif
        dir.execute(HIP_OC_COMPILER, 
            params + " " + filename + " -o " + hsaco_file
        );
    }

    hipModule_t raw_m;
    auto status = hipModuleLoad(&raw_m, hsaco_file.c_str());
    hipModulePtr m{raw_m};
    if (status != hipSuccess) MIOPEN_THROW_HIP_STATUS(status, "Failed creating module");
    return m;
}

HIPOCProgram::HIPOCProgram()
{}
HIPOCProgram::HIPOCProgram(const std::string &program_name, std::string params, bool is_kernel_str)
{
    this->module = CreateModule(program_name, params, is_kernel_str);
    this->name = program_name;
}

} // namespace miopen
