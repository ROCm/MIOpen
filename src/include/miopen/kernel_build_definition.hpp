/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/kernel.hpp>
#include <miopen/kernel_info.hpp>

#include <boost/any.hpp>

#include <functional>
#include <tuple>

namespace miopen {
namespace solver {

struct BuildProgramResult
{
    Program program;
    std::vector<std::size_t> l_wk;
    std::vector<std::size_t> g_wk;
};

using BuildParametersStringifier = std::function<std::string(const boost::any& parameters)>;

using MetadataSource =
    std::function<void(const boost::any& parameters, BuildProgramResult& target)>;

using ProgramBuilder =
    std::function<BuildProgramResult(const Handle&,
                                     const std::string& kernel_file,
                                     const boost::optional<std::string>& kernel_src,
                                     const boost::any& parameters)>;

using ProgramCache =
    std::function<BuildProgramResult(const Handle&,
                                     const std::string& kernel_file,
                                     const boost::optional<std::string>& kernel_src,
                                     const boost::any& parameters,
                                     const BuildParametersStringifier& stringifier,
                                     const MetadataSource& metadata_source,
                                     const std::function<BuildProgramResult()>& builder)>;

/// Implies that parameters are KernelInfo.
BuildProgramResult DefaultBuildProgram(const Handle& handle,
                                       const std::string& kernel_file,
                                       const boost::optional<std::string>& kernel_src,
                                       const boost::any& parameters);

BuildProgramResult BinaryProgramCache(const Handle& handle,
                                      const std::string& kernel_file,
                                      const boost::optional<std::string>& kernel_src,
                                      const boost::any& parameters,
                                      const BuildParametersStringifier& stringifier,
                                      const MetadataSource& metadata_source,
                                      const std::function<BuildProgramResult()>& builder);

class KernelBuildDefinition
{
public:
    std::string kernel_file;
    std::string kernel_name;
    boost::optional<std::string> kernel_src;
    ProgramBuilder builder;
    ProgramCache cache;
    BuildParametersStringifier stringifier;
    MetadataSource metadata_source;
    boost::any build_parameters;

    KernelBuildDefinition(
        std::string kernel_file_,
        std::string kernel_name_,
        boost::optional<std::string> kernel_src_,
        ProgramBuilder builder_,
        ProgramCache cache_,
        BuildParametersStringifier stringifier_, /// Should identify parameter set uniquely. Used
                                                 /// for program binary caching.
        MetadataSource metadata_source_,         /// For cases when program is read from cache
        boost::any build_parameters_)
        : kernel_file(std::move(kernel_file_)),
          kernel_name(std::move(kernel_name_)),
          kernel_src(std::move(kernel_src_)),
          builder(std::move(builder_)),
          cache(std::move(cache_)),
          stringifier(std::move(stringifier_)),
          metadata_source(metadata_source_),
          build_parameters(std::move(build_parameters_))
    {
    }

    KernelBuildDefinition(KernelInfo kernel_info)
        : kernel_file(kernel_info.kernel_file),
          kernel_name(kernel_info.kernel_name),
          builder(DefaultBuildProgram),
          cache(BinaryProgramCache),
          stringifier([](const boost::any& params) {
              return boost::any_cast<const KernelInfo&>(params).comp_options;
          }),
          metadata_source([](const boost::any& params, BuildProgramResult& target) {
              const auto& ki = boost::any_cast<const KernelInfo&>(params);
              target.l_wk    = ki.l_wk;
              target.g_wk    = ki.g_wk;
          }),
          build_parameters(std::move(kernel_info))
    {
    }

    BuildProgramResult operator()(const Handle& handle) const;
    BuildProgramResult operator()(const Program& program) const;
};

} // namespace solver
} // namespace miopen
