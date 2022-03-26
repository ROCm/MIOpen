/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/kernel_build_definition.hpp>

#include <miopen/binary_cache.hpp>
#include <miopen/handle.hpp>
#include <miopen/load_file.hpp>
#include <miopen/timer.hpp>

#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

namespace miopen {

namespace solver {

namespace {

// Back-end abstractions

Program LoadProgram(const Handle& handle,
                    const std::string& kernel_file,
                    const std::string& comp_options,
                    const boost::optional<std::string>& kernel_src = {});
Program
LoadCachedProgram(const Handle& handle, const std::string& program_name, const std::string& hsaco);
} // namespace

BuildProgramResult DefaultBuildProgram(const Handle& handle,
                                       const std::string& kernel_file,
                                       const boost::optional<std::string>& kernel_src,
                                       const boost::any& parameters)
{
    const auto& kernel_info = boost::any_cast<const KernelInfo&>(parameters);
    auto program = LoadProgram(handle, kernel_file, kernel_info.comp_options, kernel_src);

    return {
        std::move(program),
        kernel_info.l_wk,
        kernel_info.g_wk,
    };
}

BuildProgramResult BinaryProgramCache(const Handle& handle,
                                      const std::string& kernel_file,
                                      const boost::optional<std::string>& kernel_src,
                                      const boost::any& parameters,
                                      const BuildParametersStringifier& stringifier,
                                      const MetadataSource& metadata_source,
                                      const std::function<BuildProgramResult()>& builder)
{
    const auto parameters_str = stringifier(parameters);

    auto hsaco = miopen::LoadBinary(handle.GetTargetProperties(),
                                    handle.GetMaxComputeUnits(),
                                    kernel_src.is_initialized() ? kernel_src.value() : kernel_file,
                                    parameters_str,
                                    kernel_src.is_initialized());

    if(!hsaco.empty())
    {
        auto result    = BuildProgramResult{};
        result.program = LoadCachedProgram(handle, kernel_file, hsaco);
        metadata_source(parameters, result);
        return result;
    }

    auto built = builder();

// Save to cache
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
    auto binary = std::string{};
#if MIOPEN_BACKEND_OPENCL
    {
        auto tmp = ClProgramPtr(built.program.get());
        miopen::GetProgramBinary(tmp, binary);
        tmp.release(); // hack for API compatibiliy.
    }
#else
    binary = built.program.IsCodeObjectInMemory()
                 ? built.program.GetCodeObjectBlob()
                 : miopen::LoadFile(built.program.GetCodeObjectPathname().string());
#endif

    miopen::SaveBinary(binary,
                       handle.GetTargetProperties(),
                       handle.GetMaxComputeUnits(),
                       kernel_src.is_initialized() ? kernel_src.value() : kernel_file,
                       parameters_str,
                       kernel_src.is_initialized());
#else
    auto path = miopen::GetCachePath(false) / boost::filesystem::unique_path();

#if MIOPEN_BACKEND_OPENCL
    miopen::SaveProgramBinary(built.program, path.string());
#else
    if(program.IsCodeObjectInMemory())
        miopen::WriteFile(built.program.GetCodeObjectBlob(), path);
    else
        boost::filesystem::copy_file(built.program.GetCodeObjectPathname(), path);
#endif

    miopen::SaveBinary(path.string(),
                       handle.GetTargetProperties(),
                       kernel_src.is_initialized() ? kernel_src.value() : kernel_file,
                       parameters_str,
                       kernel_src.is_initialized());
#endif

#if MIOPEN_BACKEND_HIP
    built.program.FreeCodeObjectFileStorage();
#endif

    return built;
}

BuildProgramResult KernelBuildDefinition::operator()(const Handle& handle) const
{
    const auto builder_ = [this, &handle]() {
        CompileTimer ct;
        auto ret = builder(handle, kernel_file, kernel_src, build_parameters);
        ct.Log("Kernel", kernel_file);
        return ret;
    };

    return cache(
        handle, kernel_file, kernel_src, build_parameters, stringifier, metadata_source, builder_);
}

BuildProgramResult KernelBuildDefinition::operator()(const Program& program) const
{
    auto result    = BuildProgramResult{};
    result.program = program;
    metadata_source(build_parameters, result);
    return result;
}

namespace {

// Back-end abstractions

Program LoadProgram(const Handle& handle,
                    const std::string& kernel_file,
                    const std::string& comp_options,
                    const boost::optional<std::string>& kernel_src)
{
#if MIOPEN_BACKEND_OPENCL
    return miopen::LoadProgram(miopen::GetContext(handle.GetStream()),
                               miopen::GetDevice(handle.GetStream()),
                               handle.GetTargetProperties(),
                               kernel_file,
                               comp_options,
                               kernel_src.is_initialized(),
                               kernel_src.get_value_or(""));
#else
    return HIPOCProgram{kernel_file,
                        comp_options,
                        kernel_src.is_initialized(),
                        handle.GetTargetProperties(),
                        kernel_src.get_value_or("")};
#endif
}

Program
LoadCachedProgram(const Handle& handle, const std::string& program_name, const std::string& hsaco)
{
#if MIOPEN_BACKEND_OPENCL
    std::ignore = program_name;
    return LoadBinaryProgram(miopen::GetContext(handle.GetStream()),
                             miopen::GetDevice(handle.GetStream()),
#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
                             hsaco);
#else
                             miopen::LoadFile(hsaco));
#endif
#else
    std::ignore = handle;
    return HIPOCProgram{program_name, hsaco};
#endif
}

} // namespace

} // namespace solver
} // namespace miopen
