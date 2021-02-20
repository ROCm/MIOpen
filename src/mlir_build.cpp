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

#include <mlir-miopen-lib.hpp>

#include <fstream>

namespace miopen {

/// Generates HIP source, header and options for HIP compiler.
/// Writes HIP source and header into output directory.
///
/// @param[in]  tmp_dir  Output directory.
/// @param[in]  filename After stemming extension, used as a basename for the HIP files written.
/// @param[in]  params   Options for MLIR generator.
/// @param[out] cflags   Build options for HIP compiler.
/// @param[out] ofile    Name of output .cpp file (with path).
static void MlirGenerateSourcesForHipBuild(boost::optional<TmpDir>& tmp_dir,
                                           const std::string& filename,
                                           const std::string& params,
                                           std::string& cflags,
                                           std::string& ofile)
{
    const auto input_file      = tmp_dir->path / filename;
    const auto input_file_base = (tmp_dir->path / input_file.stem()).string();

    MIOPEN_LOG_I2(input_file.string() << ", options: '" << params << "'");

    mlir::MlirHandle handle = mlir::CreateMlirHandle(params.c_str());
    mlir::MlirLowerCpp(handle);

    ofile = input_file_base + ".cpp";

    const std::string source = mlir::MlirGenIgemmSource(handle);
    std::ofstream source_file(ofile);
    if(source.empty())
        MIOPEN_THROW("mlir::MlirGenIgemmSource() failed with" + filename);
    source_file << source;

    const std::string header = mlir::MlirGenIgemmHeader(handle);
    std::ofstream header_file(input_file_base + ".hpp");
    if(header.empty())
        MIOPEN_THROW("mlir::MlirGenIgemmHeader() failed with" + filename);
    header_file << header;

    // Get mlir kernel compilation flags.
    cflags = mlir::MlirGenIgemmCflags(handle);
    if(cflags.empty())
        MIOPEN_THROW("mlir::MlirGenIgemmCflags() failed with" + filename);

    mlir::DestroyMlirHandle(handle);

    ///\todo This smells:
    cflags     = cflags.substr(cflags.find("\n") + 1); // Skip first line.
    size_t pos = cflags.find("\n");                    // Skip end of line.
    if(pos != std::string::npos)
        cflags.replace(pos, sizeof("\n"), " ");
}

boost::filesystem::path MlirBuildViaHip(boost::optional<TmpDir>& tmp_dir,
                                        const std::string& filename,
                                        const std::string& src,
                                        const std::string& params,
                                        const TargetProperties& target)
{
    std::string hip_options;
    std::string hip_filename;
    MlirGenerateSourcesForHipBuild(tmp_dir, filename, params, hip_options, hip_filename);
    // Workaround: We can't pass HIP build options from BuildCodeObject(),
    // so let's care about warnings here.
    // Workaround: The MLIR-produced HIP code causes build warnings,
    // let's shut'em unconditionally.
    hip_options += " -Wno-everything";
    // HipBuild requires name without path.
    hip_filename = boost::filesystem::path(hip_filename).filename().string();
    return HipBuild(tmp_dir, hip_filename, src, hip_options, target, true);
}

} // namespace miopen
