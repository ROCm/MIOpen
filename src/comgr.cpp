/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/stringutils.hpp>
#include <amd_comgr.h>
#include <algorithm>
#include <cstddef>
#include <vector>

/// FIXME Rework debug stuff.
#define DEBUG_DETAILED_LOG 0
#define DEBUG_CORRUPT_SOURCE 1
#define COMPILER_LC 1

#define EC_FAILED (status != AMD_COMGR_STATUS_SUCCESS)

#define EC(expr)                                                                     \
    do                                                                               \
    {                                                                                \
        status = (expr);                                                             \
        if(EC_FAILED)                                                                \
        {                                                                            \
            const char* reason;                                                      \
            if(AMD_COMGR_STATUS_SUCCESS != amd_comgr_status_string(status, &reason)) \
                reason = "UNKNOWN";                                                  \
            MIOPEN_LOG_E("\'" #expr "\' :" << std::string(reason));                  \
        }                                                                            \
        else                                                                         \
        {                                                                            \
            MIOPEN_LOG_I("Ok \'" #expr "\'");                                        \
        }                                                                            \
    } while(false)

#define EC_BREAK(expr) \
    EC(expr);          \
    if(EC_FAILED)      \
        break;         \
    else               \
    (void)false

namespace miopen {
namespace comgr {

#if DEBUG_DETAILED_LOG
static bool PrintVersion()
{
    std::size_t major = 0;
    std::size_t minor = 0;
    amd_comgr_get_version(&major, &minor);
    MIOPEN_LOG_I("comgr v." << major << '.' << minor);
    return true;
}
static bool once = PrintVersion(); /// FIXME remove this
#endif

using OptionList = std::vector<std::string>;

/// Compiler implementation-specific functionality (compiler abstraction layer).
namespace compiler {

#if COMPILER_LC
namespace lc {
#define OCL_EARLY_INLINE 1

static void AddOclCompilerOptions(OptionList& list)
{
#if OCL_EARLY_INLINE
    list.push_back("-mllvm");
    list.push_back("-amdgpu-early-inline-all");
#endif
    list.push_back("-mllvm");
    list.push_back("-amdgpu-prelink");
    list.push_back("-mwavefrontsize64");
}

/// These are produced for offline compiler and not necessary at least
/// (or even can be harmful) for building via comgr layer.
///
/// \todo Produce proper options in, er, proper places, and get rid of this.
static void RemoveSuperfluousOptions(OptionList& list)
{
    list.erase(remove_if(list.begin(),
                         list.end(),
                         [&](const auto& option) { return StartsWith(option, "-mcpu="); }),
               list.end());
}

/// \todo Get list of supported isa names from comgr and select.
static std::string GetIsaName(const std::string& device)
{
    return {"amdgcn-amd-amdhsa--" + device};
}

/// \todo Handle "-cl-fp32-correctly-rounded-divide-sqrt".

} // namespace lc
#undef OCL_EARLY_INLINE
#endif // COMPILER_LC

} // namespace compiler

#if DEBUG_DETAILED_LOG
static void LogOptions(const char* options[], size_t count)
{
    if(miopen::IsLogging(miopen::LoggingLevel::Info))
    {
        std::ostringstream oss;
        for(std::size_t i = 0; i < count; ++i)
            oss << options[i] << '\t';
        MIOPEN_LOG_I(oss.str());
    }
}
#endif

static std::string GetLog(amd_comgr_data_set_t dataset)
{
    amd_comgr_status_t status;
    std::string log;
    amd_comgr_data_t data;
    bool data_object_obtained = false;

    do
    {
        /// Assumption: the log is the the first in the dataset.
        /// This is not specified in comgr API.
        /// Let's follow the KISS principle for now; it works.
        ///
        /// \todo Clarify API and update implementation.
        std::size_t count;
        EC_BREAK(amd_comgr_action_data_count(dataset, AMD_COMGR_DATA_KIND_LOG, &count));
        if(count < 1)
            break;

        EC_BREAK(amd_comgr_action_data_get_data(dataset, AMD_COMGR_DATA_KIND_LOG, 0, &data));
        data_object_obtained = true;

        std::size_t size;
        EC_BREAK(amd_comgr_get_data(data, &size, nullptr));
        if(size < 1)
            break;
        std::vector<char> buffer(size + 1);
        EC(amd_comgr_get_data(data, &size, &buffer[0]));
        buffer[size] = ('\0');
        log          = std::string(&buffer[0], size + 1);

    } while(false);

    if(EC_FAILED)
        log = "Failed to get log from comgr";
    else if(log.empty())
        log = "comgr log empty";
    else
        ; // nop

    // Cleanup
    if(data_object_obtained)
        EC(amd_comgr_release_data(data));

    return log;
}

static amd_comgr_status_t DatasetAddData(const std::string& name,
                                         const std::string& content,
                                         const amd_comgr_data_kind_t type,
                                         const amd_comgr_data_set_t dataset)
{
    amd_comgr_data_t handle;
    amd_comgr_status_t status;

    EC(amd_comgr_create_data(type, &handle));
    if(EC_FAILED)
        return status;

    do
    {
        MIOPEN_LOG_I(name << ' ' << content.size() << " bytes");
        EC_BREAK(amd_comgr_set_data_name(handle, name.c_str()));
#if DEBUG_DETAILED_LOG
        if(miopen::IsLogging(miopen::LoggingLevel::Info) && type == AMD_COMGR_DATA_KIND_SOURCE)
        {
            const auto text_length = (content.size() > 256) ? 256 : content.size();
            const std::string text(content, 0, text_length);
            MIOPEN_LOG_I(text);
        }
#endif
#if DEBUG_CORRUPT_SOURCE
        std::string bad("int xxx = 1 yyy = 2;\n");
        bad += content;
        EC_BREAK(amd_comgr_set_data(handle, bad.size(), bad.data()));
#else
        EC_BREAK(amd_comgr_set_data(handle, content.size(), content.data()));
#endif
        EC_BREAK(amd_comgr_data_set_add(dataset, handle));
    } while(false);

    EC(amd_comgr_release_data(handle)); // Will be finally destroyed together with dataset.
    return status;
}

void BuildOcl(const std::string& name,
              const std::string& text,
              const std::string& options,
              const std::string& device,
              const std::string& binary)
{
    (void)binary;

    amd_comgr_status_t status;
    amd_comgr_data_set_t inputs;
    bool inputs_created = false;
    amd_comgr_action_info_t action;
    bool action_created = false;
    amd_comgr_data_set_t pch;
    bool pch_created = false;
    amd_comgr_data_set_t src2bc;
    bool src2bc_created = false;
    do
    {
        EC_BREAK(amd_comgr_create_data_set(&inputs));
        inputs_created = true;

        EC_BREAK(DatasetAddData(name, text, AMD_COMGR_DATA_KIND_SOURCE, inputs));

        EC_BREAK(amd_comgr_create_action_info(&action));
        action_created = true;
        EC_BREAK(amd_comgr_action_info_set_language(action, AMD_COMGR_LANGUAGE_OPENCL_1_2));
        const auto isaName = compiler::lc::GetIsaName(device);
        MIOPEN_LOG_I(isaName);
        EC_BREAK(amd_comgr_action_info_set_isa_name(action, isaName.c_str()));
        EC_BREAK(amd_comgr_action_info_set_logging(action, true));

        auto optList = SplitSpaceSeparated(options);
        compiler::lc::RemoveSuperfluousOptions(optList);
        compiler::lc::AddOclCompilerOptions(optList);
        {
            std::vector<const char*> vp;
            vp.reserve(optList.size());
            for(auto& opt : optList) // cppcheck-suppress useStlAlgorithm
                vp.push_back(opt.c_str());
#if DEBUG_DETAILED_LOG
            LogOptions(vp.data(), vp.size());
#endif
            EC_BREAK(amd_comgr_action_info_set_option_list(action, vp.data(), vp.size()));
        }

        EC_BREAK(amd_comgr_create_data_set(&pch));
        pch_created = true;
        EC(amd_comgr_do_action(AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS, action, inputs, pch));
        if(EC_FAILED)
        {
            MIOPEN_LOG_W("comgr: " << GetLog(pch));
            break;
        }

        EC_BREAK(amd_comgr_create_data_set(&src2bc));
        src2bc_created = true;
        EC(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, action, pch, src2bc));
        if(EC_FAILED)
        {
            MIOPEN_LOG_W("comgr: " << GetLog(src2bc));
            break;
        }

    } while(false);

    { // If cleanup fails, then do not throw, just issue error messages.
        const auto status_save = status;
        if(src2bc_created)
            EC(amd_comgr_destroy_data_set(src2bc));
        if(pch_created)
            EC(amd_comgr_destroy_data_set(pch));
        if(action_created)
            EC(amd_comgr_destroy_action_info(action));
        if(inputs_created)
            EC(amd_comgr_destroy_data_set(inputs));
        status = status_save;
    }

    if(EC_FAILED)
    {
        MIOPEN_THROW("comgr::BuildOcl");
    }
}

} // namespace comgr
} // namespace miopen
