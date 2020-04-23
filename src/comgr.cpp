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
#include <exception>
#include <cstddef>
#include <cstring>
#include <tuple> // std::ignore
#include <vector>

/// FIXME Rework debug stuff.
#define DEBUG_DETAILED_LOG 0
#define DEBUG_CORRUPT_SOURCE 0
#define COMPILER_LC 1

#define EC_BASE(expr, info, expr2)                                                              \
    do                                                                                          \
    {                                                                                           \
        const amd_comgr_status_t status = (expr);                                               \
        if(status != AMD_COMGR_STATUS_SUCCESS)                                                  \
        {                                                                                       \
            MIOPEN_LOG_E("\'" #expr "\' " << to_string(info) << ": " << GetStatusText(status)); \
            (expr2);                                                                            \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            MIOPEN_LOG_I("Ok \'" #expr "\' " << to_string(info));                               \
        }                                                                                       \
    } while(false)

#define EC(expr) EC_BASE(expr, NoInfo, (void)0)
#define EC_THROW(expr) EC_BASE(expr, NoInfo, Throw(status))
#define ECI_THROW(expr, info) EC_BASE(expr, info, Throw(status))
#define ECI_THROW_MSG(expr, info, msg) EC_BASE(expr, info, Throw(status, (msg)))

namespace miopen {
namespace comgr {

using OptionList = std::vector<std::string>;

/// Compiler implementation-specific functionality (compiler abstraction layer).
namespace compiler {

#if COMPILER_LC
namespace lc {
#define OCL_EARLY_INLINE 1

/// gfx1000+
/// - WGP mode: not enabled.
/// - WAVE32 mode: not enabled.

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

struct EcInfoNone
{
};
static const EcInfoNone NoInfo;
static inline std::string to_string(const EcInfoNone&) { return {}; }
static inline std::string to_string(const std::string& v) { return {v}; }
static inline std::string to_string(const bool& v) { return v ? "true" : "false"; }
static inline auto to_string(const std::size_t& v) { return std::to_string(v); }

#define CASE_TO_STRING(id) \
    case id: return #id

static std::string to_string(const amd_comgr_language_t val)
{
    switch(val)
    {
        CASE_TO_STRING(AMD_COMGR_LANGUAGE_NONE);
        CASE_TO_STRING(AMD_COMGR_LANGUAGE_OPENCL_1_2);
        CASE_TO_STRING(AMD_COMGR_LANGUAGE_OPENCL_2_0);
        CASE_TO_STRING(AMD_COMGR_LANGUAGE_HC);
        CASE_TO_STRING(AMD_COMGR_LANGUAGE_HIP);
    }
    return "<Unknown language>";
}

static std::string to_string(const amd_comgr_data_kind_t val)
{
    switch(val)
    {
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_UNDEF);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_SOURCE);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_INCLUDE);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_DIAGNOSTIC);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_LOG);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_BC);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_RELOCATABLE);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_EXECUTABLE);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_BYTES);
        CASE_TO_STRING(AMD_COMGR_DATA_KIND_FATBIN);
    }
    return "<Unknown data kind>";
}

static std::string to_string(const amd_comgr_action_kind_t val)
{
    switch(val)
    {
        CASE_TO_STRING(AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR);
        CASE_TO_STRING(AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS);
        CASE_TO_STRING(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC);
        CASE_TO_STRING(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES);
        CASE_TO_STRING(AMD_COMGR_ACTION_LINK_BC_TO_BC);
        CASE_TO_STRING(AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC);
        CASE_TO_STRING(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE);
        CASE_TO_STRING(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY);
        CASE_TO_STRING(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE);
        CASE_TO_STRING(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE);
        CASE_TO_STRING(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE);
        CASE_TO_STRING(AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE);
        CASE_TO_STRING(AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE);
        CASE_TO_STRING(AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE);
        CASE_TO_STRING(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN);
    }
    return "<Unknown action kind>";
}

static bool PrintVersion()
{
    std::size_t major = 0;
    std::size_t minor = 0;
    (void)amd_comgr_get_version(&major, &minor);
    MIOPEN_LOG_I("comgr v." << major << '.' << minor);
    return true;
}

static std::string GetStatusText(const amd_comgr_status_t status)
{
    const char* reason = nullptr;
    if(AMD_COMGR_STATUS_SUCCESS != amd_comgr_status_string(status, &reason))
        reason = "<Unknown>";
    return std::string(reason) + " (" + std::to_string(static_cast<int>(status)) + ')';
}

static void LogOptions(const char* options[], size_t count)
{
#if DEBUG_DETAILED_LOG
    if(miopen::IsLogging(miopen::LoggingLevel::Info))
    {
        std::ostringstream oss;
        for(std::size_t i = 0; i < count; ++i)
            oss << options[i] << '\t';
        MIOPEN_LOG_I(oss.str());
    }
#else
    std::ignore = options;
    std::ignore = count;
#endif
}

class Dataset;
static std::string GetLog(const Dataset& dataset, bool catch_comgr_exceptions = false);

struct ComgrError : std::exception
{
    amd_comgr_status_t status;
    std::string text;

    ComgrError(const amd_comgr_status_t s) : status(s) {}
    ComgrError(const amd_comgr_status_t s, const std::string& t) : status(s), text(t) {}
    const char* what() const noexcept override { return text.c_str(); }
};

struct ComgrOwner
{
    ComgrOwner(const ComgrOwner&) = delete;

    protected:
    ComgrOwner() {}
    ComgrOwner(ComgrOwner&&) = default;
    [[noreturn]] void Throw(const amd_comgr_status_t s) const { throw ComgrError{s}; }
    [[noreturn]] void Throw(const amd_comgr_status_t s, const std::string& text) const
    {
        throw ComgrError{s, text};
    }
};

class Data : ComgrOwner
{
    amd_comgr_data_t handle = {0};
    friend class Dataset; // for GetData
    Data(amd_comgr_data_t h) : handle(h) {}

    public:
    Data(amd_comgr_data_kind_t kind) { ECI_THROW(amd_comgr_create_data(kind, &handle), kind); }
    Data(Data&&) = default;
    ~Data() { EC(amd_comgr_release_data(handle)); }
    auto operator()() const { return handle; }
    void SetName(const std::string& s) const
    {
        ECI_THROW(amd_comgr_set_data_name(handle, s.c_str()), s);
    }
    void SetBytes(const std::string& bytes) const
    {
        ECI_THROW(amd_comgr_set_data(handle, bytes.size(), bytes.data()), bytes.size());
    }

    private:
    std::size_t GetSize() const
    {
        std::size_t sz;
        EC_THROW(amd_comgr_get_data(handle, &sz, nullptr));
        return sz;
    }
    std::size_t GetBytes(std::vector<char>& bytes) const
    {
        std::size_t sz = GetSize();
        bytes.resize(sz);
        ECI_THROW(amd_comgr_get_data(handle, &sz, &bytes[0]), sz);
        return sz;
    }

    public:
    std::string GetString() const
    {
        std::vector<char> bytes;
        const auto sz = GetBytes(bytes);
        return {&bytes[0], sz};
    }
};

class Dataset : ComgrOwner
{
    amd_comgr_data_set_t handle = {0};

    public:
    Dataset() { EC_THROW(amd_comgr_create_data_set(&handle)); }
    ~Dataset() { EC(amd_comgr_destroy_data_set(handle)); }
    auto operator()() const { return handle; }
    void AddData(const Data& d) const { EC_THROW(amd_comgr_data_set_add(handle, d())); }
    size_t GetDataCount(const amd_comgr_data_kind_t kind) const
    {
        std::size_t count = 0;
        ECI_THROW(amd_comgr_action_data_count(handle, kind, &count), kind);
        return count;
    }
    Data GetData(const amd_comgr_data_kind_t kind, const std::size_t index) const
    {
        amd_comgr_data_t d;
        ECI_THROW(amd_comgr_action_data_get_data(handle, kind, index, &d), kind);
        return {d};
    }
};

class ActionInfo : ComgrOwner
{
    amd_comgr_action_info_t handle = {0};

    public:
    ActionInfo() { EC_THROW(amd_comgr_create_action_info(&handle)); }
    ~ActionInfo() { EC(amd_comgr_destroy_action_info(handle)); }
    auto operator()() const { return handle; }
    void SetLanguage(const amd_comgr_language_t language) const
    {
        ECI_THROW(amd_comgr_action_info_set_language(handle, language), language);
    }
    void SetIsaName(const std::string& isa) const
    {
        ECI_THROW_MSG(amd_comgr_action_info_set_isa_name(handle, isa.c_str()), isa, isa);
    }
    void SetLogging(const bool state) const
    {
        ECI_THROW(amd_comgr_action_info_set_logging(handle, state), state);
    }
    void SetOptionList(const std::vector<std::string>& options) const
    {
        std::vector<const char*> vp;
        vp.reserve(options.size());
        for(auto& opt : options) // cppcheck-suppress useStlAlgorithm
            vp.push_back(opt.c_str());
        LogOptions(vp.data(), vp.size());
        ECI_THROW(amd_comgr_action_info_set_option_list(handle, vp.data(), vp.size()), vp.size());
    }
    void Do(const amd_comgr_action_kind_t kind, const Dataset& in, const Dataset& out) const
    {
        ECI_THROW_MSG(amd_comgr_do_action(kind, handle, in(), out()), kind, GetLog(out, true));
#if DEBUG_DETAILED_LOG
        const auto log = GetLog(out);
        if(!log.empty())
            MIOPEN_LOG_I(GetLog(out));
#endif
    }
};

/// If called from the context of comgr error handling:
/// - We do not allow the comgr-induced exceptions to escape.
/// -
/// when called the context of comgr error handling.
static std::string GetLog(const Dataset& dataset, const bool comgr_error_handling)
{
    std::string text;
    try
    {
        /// Assumption: the log is the the first in the dataset.
        /// This is not specified in comgr API.
        /// Let's follow the KISS principle for now; it works.
        ///
        /// \todo Clarify API and update implementation.
        const auto count = dataset.GetDataCount(AMD_COMGR_DATA_KIND_LOG);
        if(count < 1)
            return {comgr_error_handling ? "comgr warning: error log not found" : ""};

        const auto data = dataset.GetData(AMD_COMGR_DATA_KIND_LOG, 0);
        text            = data.GetString();
        if(text.empty())
            return {comgr_error_handling ? "comgr info: error log empty" : ""};
    }
    catch(ComgrError&)
    {
        if(comgr_error_handling)
            return {"comgr error: failed to get error log"};
        throw;
    }
    return text;
}

static void DatasetAddData(const Dataset& dataset,
                           const std::string& name,
                           const std::string& content,
                           const amd_comgr_data_kind_t type)
{
    const Data d(type);
    MIOPEN_LOG_I(name << ' ' << content.size() << " bytes");
    d.SetName(name);
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
    d.SetBytes(bad);
#else
    d.SetBytes(content);
#endif
    dataset.AddData(d);
}

void BuildOcl(const std::string& name,
              const std::string& text,
              const std::string& options,
              const std::string& device,
              const std::string& binary)
{
    (void)binary;
    static const auto once = PrintVersion();
    std::ignore            = once;

    try
    {
        const Dataset inputs;
        DatasetAddData(inputs, name, text, AMD_COMGR_DATA_KIND_SOURCE);
        const ActionInfo action;
        action.SetLanguage(AMD_COMGR_LANGUAGE_OPENCL_1_2);
        const auto isaName = compiler::lc::GetIsaName(device);
        MIOPEN_LOG_I(isaName); // FIXME
        action.SetIsaName(isaName);
        action.SetLogging(true);

        auto optList = SplitSpaceSeparated(options);
        compiler::lc::RemoveSuperfluousOptions(optList);
        compiler::lc::AddOclCompilerOptions(optList);
        action.SetOptionList(optList);

        const Dataset pch;
        action.Do(AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS, inputs, pch);
        const Dataset src2bc;
        action.Do(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, pch, src2bc);

        OptionList optLink;
        optLink.push_back("wavefrontsize64");
        for(const auto& opt : optList)
        {
            if(opt == "-cl-fp32-correctly-rounded-divide-sqrt")
                optLink.push_back("correctly_rounded_sqrt");
            else if(opt == "-cl-denorms-are-zero")
                optLink.push_back("daz_opt");
            else if(opt == "-cl-finite-math-only" || opt == "cl-fast-relaxed-math")
                optLink.push_back("finite_only");
            else if(opt == "-cl-unsafe-math-optimizations" || opt == "-cl-fast-relaxed-math")
                optLink.push_back("unsafe_math");
            else
            {
            } // nop
        }
        action.SetOptionList(optLink);
        const Dataset withDevLibs;
        action.Do(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES, src2bc, withDevLibs);
        const Dataset linkedBc2bc;
        action.Do(AMD_COMGR_ACTION_LINK_BC_TO_BC, withDevLibs, linkedBc2bc);

    }
    catch(ComgrError& ex)
    {
        MIOPEN_LOG_E("comgr status = " << GetStatusText(ex.status));
        if(!ex.text.empty())
            MIOPEN_LOG_W(ex.text);
        MIOPEN_THROW(MIOPEN_GET_FN_NAME() + ": comgr status = " + GetStatusText(ex.status));
    }
}

} // namespace comgr
} // namespace miopen
