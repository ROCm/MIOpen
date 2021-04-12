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

#include <miopen/config.h>
#include <miopen/hip_build_utils.hpp>
#include <miopen/logger.hpp>
#include <miopen/mlir_build.hpp>

#include <Miir.h>

#include <fstream>

namespace miopen {
// Anonymous namespace
namespace {
/// Destroys handle in case of exception.
class AutoMiirHandle
{
    MiirHandle handle;

    public:
    AutoMiirHandle(const std::string& options) : handle(miirCreateHandle(options.c_str())) {}
    ~AutoMiirHandle() { miirDestroyHandle(handle); }
    MiirHandle operator()() { return handle; }
};

void check_miir_error(MiirStatus status, const std::string& miir_fn_name)
{
    switch(status)
    {
    case MIIR_SUCCESS: return;
    case MIIR_INVALID_PARAM: MIOPEN_THROW(miir_fn_name + " MIIR_INVALID_PARAM"); break;
    case MIIR_INVALID_MODULE: MIOPEN_THROW(miir_fn_name + " MIIR_INVALID_MODULE"); break;
    case MIIR_BUILD_FAILURE: MIOPEN_THROW(miir_fn_name + " MIIR_BUILD_FAILURE"); break;
    default: MIOPEN_THROW(miir_fn_name + " <UNKNOWN ERROR>");
    }
}
} // namespace
/// Generates HIP source, header and options for HIP compiler.
/// Writes HIP source and header into output directory.
///
/// @param[in]  tmp_dir  Output directory.
/// @param[in]  filename After stemming extension, used as a basename for the HIP files written.
/// @param[in]  params   Options for MLIR generator.
/// @param[out] cflags   Build options for HIP compiler.
/// @param[out] cpp_filename  Name of output .cpp file (with path).
static void MiirGenerateSourcesForHipBuild(const boost::optional<TmpDir>& tmp_dir,
                                           const std::string& filename,
                                           const std::string& params,
                                           std::string& cflags,
                                           std::string& cpp_filename)
{
    static const auto this_fn_name = MIOPEN_GET_FN_NAME();
    auto throw_if_error            = [&](const std::string& mlir_fn_name,
                              const std::string& generated,
                              const std::ofstream* const ofs  = nullptr,
                              const std::string& ofs_filename = {}) {
        if(generated.empty())
            MIOPEN_THROW("In " + this_fn_name + ": " + mlir_fn_name + "() failed with" + filename);
        if(ofs != nullptr && !ofs->is_open())
            MIOPEN_THROW("In " + this_fn_name + ": " + "Failed to open " + ofs_filename);
    };

    const auto input_file      = tmp_dir->path / filename;
    const auto input_file_base = (tmp_dir->path / input_file.stem()).string();

    MIOPEN_LOG_I2(input_file.string() << ", options: '" << params << "'");

    AutoMiirHandle handle(params);
    miirLowerCpp(handle());

    cpp_filename = input_file_base + ".cpp";
    const auto hpp_filename(input_file_base + ".hpp");

    const std::string cpp_text = miirGenIgemmSource(handle());
    std::ofstream cpp_ofs(cpp_filename);
    throw_if_error("miirGenIgemmSource", cpp_text, &cpp_ofs, cpp_filename);
    cpp_ofs << cpp_text;

    const std::string hpp_text = miirGenIgemmHeader(handle());
    std::ofstream hpp_ofs(hpp_filename);
    throw_if_error("miirGenIgemmHeader", hpp_text, &hpp_ofs, hpp_filename);
    hpp_ofs << hpp_text;

    // Get mlir kernel compilation flags.
    cflags = miirGenIgemmCflags(handle());
    throw_if_error("miirGenIgemmCflags", cflags);

    ///\todo This smells:
    cflags     = cflags.substr(cflags.find("\n") + 1); // Skip first line.
    size_t pos = cflags.find("\n");                    // Skip end of line.
    if(pos != std::string::npos)
        cflags.replace(pos, sizeof("\n"), " ");
}

boost::filesystem::path MiirBuildViaHip(boost::optional<TmpDir>& tmp_dir,
                                        const std::string& filename,
                                        const std::string& src,
                                        const std::string& params,
                                        const TargetProperties& target)
{
    std::string hip_options;
    std::string hip_filename;
    MiirGenerateSourcesForHipBuild(tmp_dir, filename, params, hip_options, hip_filename);
    // Workaround: We can't pass HIP build options from BuildCodeObject(),
    // so let's care about warnings here.
    // Workaround: The MLIR-produced HIP code causes build warnings,
    // let's shut'em unconditionally.
    hip_options += " -Wno-everything";
    // HipBuild requires name without path.
    hip_filename = boost::filesystem::path(hip_filename).filename().string();
    return HipBuild(tmp_dir, hip_filename, src, hip_options, target, true);
}

void MiirGenLaunchParams(const std::string& params, size_t& local_size, size_t& global_size)
{
    AutoMiirHandle handle(params);
    miirLowerInit();
    auto status = miirLowerTuningParams(handle());
    check_miir_error(status, "miirLowerTuningParams");
    miirGetExecutionDims(handle(), &global_size, &local_size);
    check_miir_error(status, "miirGetExecutionDims");
}

} // namespace miopen
