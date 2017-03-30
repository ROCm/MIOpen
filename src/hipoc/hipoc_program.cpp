#include <mlopen/hipoc_program.hpp>
#include <mlopen/kernel.hpp>
#include <mlopen/errors.hpp>
#include <mlopen/stringutils.hpp>

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
        std::string cmd = cd + exe + " " + args;// + " > /dev/null";
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
    // std::cerr << "Write file: " << name << std::endl;
    HIPOCProgram::FilePtr f{std::fopen(name.c_str(), "w")};
    if (std::fwrite(content.c_str(), 1, content.size(), f.get()) != content.size()) 
        MLOPEN_THROW("Failed to write to src file");
}

hipModulePtr CreateModule(const std::string& program_name, std::string params, bool is_kernel_str)
{
    std::string filename = is_kernel_str ? "tinygemm.cl" : program_name;
    tmp_dir dir{filename};
    std::string bin_file = dir.path(filename) + ".bin";
    std::string hsaco_file = dir.path(filename) + ".o";
    std::string obj_file = dir.path(filename) + ".obj";

    std::string src = is_kernel_str ? program_name : GetKernelSrc(program_name);
    if (!is_kernel_str && mlopen::EndsWith(program_name, ".so"))
    {
        WriteFile(src, hsaco_file);        
    }
    else
    {

        WriteFile(src, dir.path(filename));

#ifdef HIP_OC_FINALIZER
        // Adding the same flags / defines to aoc2 that the OCL runtime adds for calls
        // to clBuildProgram(). This has been shown to significantly affect performance.
        params +=
          " "
          "-DCL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE=2605857024 -DFP_FAST_FMA=1 "
          "-cl-denorms-are-zero -m64 -Dcl_khr_fp64=1 -Dcl_amd_fp64=1 "
          "-Dcl_khr_global_int32_base_atomics=1 -Dcl_khr_global_int32_extended_atomics=1 "
          "-Dcl_khr_local_int32_base_atomics=1 -Dcl_khr_local_int32_extended_atomics=1 "
          "-Dcl_khr_int64_base_atomics=1 -Dcl_khr_int64_extended_atomics=1 "
          "-Dcl_khr_3d_image_writes=1 -Dcl_khr_byte_addressable_store=1 -Dcl_khr_fp16=1 "
          "-Dcl_khr_gl_sharing=1 -Dcl_khr_gl_depth_images=1 "
          "-Dcl_amd_device_attribute_query=1 -Dcl_amd_vec3=1 -Dcl_amd_printf=1 "
          "-Dcl_amd_media_ops=1 -Dcl_amd_media_ops2=1 -Dcl_amd_popcnt=1 "
          "-Dcl_khr_image2d_from_buffer=1 -Dcl_khr_spir=1 -Dcl_khr_subgroups=1 "
          "-Dcl_khr_gl_event=1 -Dcl_khr_depth_images=1 -Dcl_khr_mipmap_image=1 "
          "-Dcl_khr_mipmap_image_writes=1"
          " ";
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
            filename);
        dir.execute(HIP_OC_FINALIZER,
            "-target=8:0:3 -hsail dump_0_Fiji.hsail -output=" + hsaco_file
        );
#endif

#else
        params += " -Weverything -Wno-shorten-64-to-32 -Wno-unused-macros -Wno-unused-function -Wno-sign-compare -Wno-reserved-id-macro ";
        dir.execute(HIP_OC_COMPILER, 
            "-mcpu=gfx803 " + params + " " + filename + " -o " + hsaco_file
        );
#endif
    }

    hipModule_t raw_m;
    auto status = hipModuleLoad(&raw_m, hsaco_file.c_str());
    hipModulePtr m{raw_m};
    if (status != hipSuccess) MLOPEN_THROW_HIP_STATUS(status, "Failed creating module");
    return m;
}

HIPOCProgram::HIPOCProgram()
{}
HIPOCProgram::HIPOCProgram(const std::string &program_name, std::string params, bool is_kernel_str)
{
    this->module = CreateModule(program_name, params, is_kernel_str);
    this->name = program_name;
}

}
