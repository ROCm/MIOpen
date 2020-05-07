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

#include <miopen/env.hpp>
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

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_COMGR_LOG_CALLS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_COMGR_LOG_OPTIONS)
/// Integer, set to max number of first characters
/// you would like to log onto console.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_COMGR_LOG_SOURCE_TEXT)

/// \todo see issue #1222, PR #1316
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_SRAM_EDC_DISABLED)

#define COMPILER_LC 1

#define EC_BASE(comgrcall, info, action)                                  \
    do                                                                    \
    {                                                                     \
        const amd_comgr_status_t status = (comgrcall);                    \
        if(status != AMD_COMGR_STATUS_SUCCESS)                            \
        {                                                                 \
            MIOPEN_LOG_E("\'" #comgrcall "\' " << to_string(info) << ": " \
                                               << GetStatusText(status)); \
            (action);                                                     \
        }                                                                 \
        else if(miopen::IsEnabled(MIOPEN_DEBUG_COMGR_LOG_CALLS{}))        \
            MIOPEN_LOG_I("Ok \'" #comgrcall "\' " << to_string(info));    \
    } while(false)

// Regarding EC*THROW* macros:
// Q: Whats the point in logging if it is going to throw?
// A: This prints build log (that presumably contains warning
// and error messages from compiler/linker etc) onto console,
// thus informing the *user*. MIOPEN_THROW informs the *library*
// that compilation has failed.

#define EC(comgrcall) EC_BASE(comgrcall, NoInfo, (void)0)
#define EC_THROW(comgrcall) EC_BASE(comgrcall, NoInfo, Throw(status))
#define ECI_THROW(comgrcall, info) EC_BASE(comgrcall, info, Throw(status))
#define ECI_THROW_MSG(comgrcall, info, msg) EC_BASE(comgrcall, info, Throw(status, (msg)))

namespace miopen {
namespace comgr {

using OptionList = std::vector<std::string>;

/// Compiler implementation-specific functionality
/// (minimal compiler abstraction layer).
namespace compiler {

#if COMPILER_LC
namespace lc {
#define OCL_EARLY_INLINE 1

static void AddOcl20CompilerOptions(OptionList& list)
{
    list.push_back("-cl-kernel-arg-info");
    list.push_back("-D__IMAGE_SUPPORT__=1");
    list.push_back("-D__OPENCL_VERSION__=200");
#if OCL_EARLY_INLINE
    list.push_back("-mllvm");
    list.push_back("-amdgpu-early-inline-all");
#endif
    list.push_back("-mllvm");
    list.push_back("-amdgpu-prelink");
    list.push_back("-mwavefrontsize64"); // gfx1000+ WAVE32 mode: always disabled.
    list.push_back("-mcumode");          // gfx1000+ WGP mode: always disabled.
    list.push_back("-O3");

    // It seems like these options are used only in codegen.
    // However it seems ok to pass these to compiler.
    if(!miopen::IsEnabled(MIOPEN_DEBUG_SRAM_EDC_DISABLED{}))
        list.push_back("-msram-ecc");
    else
        list.push_back("-mno-sram-ecc");
    list.push_back("-mllvm");
    list.push_back("-amdgpu-internalize-symbols");
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
    const char* const ecc_suffix = (!miopen::IsEnabled(MIOPEN_DEBUG_SRAM_EDC_DISABLED{}) &&
                                    (device == "gfx906" || device == "gfx908"))
                                       ? "+sram-ecc"
                                       : "";
    return {"amdgcn-amd-amdhsa--" + device + ecc_suffix};
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

/// \todo Request comgr to expose this stuff via API.
static std::string to_string(const amd_comgr_language_t val)
{
    std::ostringstream oss;
    MIOPEN_LOG_ENUM(oss,
                    val,
                    AMD_COMGR_LANGUAGE_NONE,
                    AMD_COMGR_LANGUAGE_OPENCL_1_2,
                    AMD_COMGR_LANGUAGE_OPENCL_2_0,
                    AMD_COMGR_LANGUAGE_HC,
                    AMD_COMGR_LANGUAGE_HIP);
    return oss.str();
}

static std::string to_string(const amd_comgr_data_kind_t val)
{
    std::ostringstream oss;
    MIOPEN_LOG_ENUM(oss,
                    val,
                    AMD_COMGR_DATA_KIND_UNDEF,
                    AMD_COMGR_DATA_KIND_SOURCE,
                    AMD_COMGR_DATA_KIND_INCLUDE,
                    AMD_COMGR_DATA_KIND_PRECOMPILED_HEADER,
                    AMD_COMGR_DATA_KIND_DIAGNOSTIC,
                    AMD_COMGR_DATA_KIND_LOG,
                    AMD_COMGR_DATA_KIND_BC,
                    AMD_COMGR_DATA_KIND_RELOCATABLE,
                    AMD_COMGR_DATA_KIND_EXECUTABLE,
                    AMD_COMGR_DATA_KIND_BYTES,
                    AMD_COMGR_DATA_KIND_FATBIN);
    return oss.str();
}

static std::string to_string(const amd_comgr_action_kind_t val)
{
    std::ostringstream oss;
    MIOPEN_LOG_ENUM(oss,
                    val,
                    AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR,
                    AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS,
                    AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC,
                    AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES,
                    AMD_COMGR_ACTION_LINK_BC_TO_BC,
                    AMD_COMGR_ACTION_OPTIMIZE_BC_TO_BC,
                    AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                    AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY,
                    AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE,
                    AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                    AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                    AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE,
                    AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE,
                    AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE,
                    AMD_COMGR_ACTION_COMPILE_SOURCE_TO_FATBIN);
    return oss.str();
}

static bool PrintVersion()
{
    std::size_t major = 0;
    std::size_t minor = 0;
    (void)amd_comgr_get_version(&major, &minor);
    MIOPEN_LOG_NQI("comgr v." << major << '.' << minor);
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
    if(miopen::IsEnabled(MIOPEN_DEBUG_COMGR_LOG_OPTIONS{}) &&
       miopen::IsLogging(miopen::LoggingLevel::Info))
    {
        std::ostringstream oss;
        for(std::size_t i = 0; i < count; ++i)
            oss << options[i] << '\t';
        MIOPEN_LOG_I(oss.str());
    }
}

class Dataset;
static std::string GetLog(const Dataset& dataset, bool comgr_error_handling = false);

// Q: Why arent we using MIOPEN_THROW? Using MIOPEN_THROW can report back the
// source line numbers where the exception was thrown. Usually we write
// a function to convert the error enum into a message and then pass that
// MIOPEN_THROW.
//
// A: These exceptions are not intended to report "normal" miopen errors.
// The main purpose is to prevent resource leakage (comgr handles)
// when compilation of the device code fails. The side functionality is to
// hold status codes and diagnostic messages received from comgr
// when build failure happens.
//
// The diagnostic messages are expected to be like the ones that
// offline compiler prints after build errors. Usually these
// contain the file/line information of the problematic device code,
// so file/line of the host code is not needed here
// (and even considered harmful).
//
// These exceptions are not allowed to escape comgr module.
//
// The comgr module can be considered as a "library within a library"
// that provides HIP backend with functionality similar to clBuildProgram().

struct ComgrError : std::exception
{
    amd_comgr_status_t status;
    std::string text;

    ComgrError(const amd_comgr_status_t s) : status(s) {}
    ComgrError(const amd_comgr_status_t s, const std::string& t) : status(s), text(t) {}
    const char* what() const noexcept override { return text.c_str(); }
};

[[noreturn]] static void Throw(const amd_comgr_status_t s) { throw ComgrError{s}; }
[[noreturn]] static void Throw(const amd_comgr_status_t s, const std::string& text)
{
    throw ComgrError{s, text};
}

struct ComgrOwner
{
    ComgrOwner(const ComgrOwner&) = delete;

    protected:
    ComgrOwner() {}
    ComgrOwner(ComgrOwner&&) = default;
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
    auto GetHandle() const { return handle; }
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

    public:
    std::size_t GetBytes(std::vector<char>& bytes) const
    {
        std::size_t sz = GetSize();
        bytes.resize(sz);
        ECI_THROW(amd_comgr_get_data(handle, &sz, &bytes[0]), sz);
        return sz;
    }
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
    auto GetHandle() const { return handle; }
    void AddData(const Data& d) const { EC_THROW(amd_comgr_data_set_add(handle, d.GetHandle())); }
    void AddData(const std::string& name,
                 const std::string& content,
                 const amd_comgr_data_kind_t type) const
    {
        const Data d(type);
        MIOPEN_LOG_I2(name << ' ' << content.size() << " bytes");
        d.SetName(name);
        d.SetBytes(content);
        AddData(d);
        const auto show_first = miopen::Value(MIOPEN_DEBUG_COMGR_LOG_SOURCE_TEXT{}, 0);
        if(show_first > 0 && miopen::IsLogging(miopen::LoggingLevel::Info) &&
           type == AMD_COMGR_DATA_KIND_SOURCE)
        {
            const auto text_length = (content.size() > show_first) ? show_first : content.size();
            const std::string text(content, 0, text_length);
            MIOPEN_LOG_I(text);
        }
    }
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
        std::transform(options.begin(), options.end(), std::back_inserter(vp), [&](auto& opt) {
            return opt.c_str();
        });
        LogOptions(vp.data(), vp.size());
        ECI_THROW(amd_comgr_action_info_set_option_list(handle, vp.data(), vp.size()), vp.size());
    }
    void Do(const amd_comgr_action_kind_t kind, const Dataset& in, const Dataset& out) const
    {
        ECI_THROW_MSG(amd_comgr_do_action(kind, handle, in.GetHandle(), out.GetHandle()),
                      kind,
                      GetLog(out, true));
        const auto log = GetLog(out);
        if(!log.empty())
            MIOPEN_LOG_I(to_string(kind) << ": " << log);
    }
};

/// If called from the context of comgr error handling:
/// - We do not allow the comgr-induced exceptions to escape.
/// - If obtaining log fails, write some diagnostics into output.
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
        // deepcode ignore EmptyThrowOutsideCatch: false positive
        throw;
        // Q: What the point in catching the error if you are just going to rethrow it?
        //
        // A: In the context of handling of build error, the function is invoked to get build log
        // from comgr. If it fails, it doesn't rethrow because this would overwrite the
        // original comgr status.
        //
        // The use case is when some build error happens (e.g. linking error) but we unable to
        // obtain log data from the dataset due to comgr error. We keep original error information
        // (albeit only status code). The user will see error message with original status code
        // plus comgr error: "failed to get error log."
        //
        // Rethrowing happens when/if this function is invoked during normal flow, i.e. when
        // there is no build errors. In such a case, the catch block does nothing and allows
        // all exceptions to escape. This would effectively stop the build.
    }
    return text;
}

void BuildOcl(const std::string& name,
              const std::string& text,
              const std::string& options,
              const std::string& device,
              std::vector<char>& binary)
{
    static const auto once = PrintVersion(); // Nice to see in the user's logs.
    std::ignore            = once;

    try
    {
        const Dataset inputs;
        inputs.AddData(name, text, AMD_COMGR_DATA_KIND_SOURCE);
        const ActionInfo action;
        action.SetLanguage(AMD_COMGR_LANGUAGE_OPENCL_2_0);
        const auto isaName = compiler::lc::GetIsaName(device);
        MIOPEN_LOG_I2(isaName);
        action.SetIsaName(isaName);
        action.SetLogging(true);

        auto optCompile = SplitSpaceSeparated(options);
        compiler::lc::RemoveSuperfluousOptions(optCompile);
        compiler::lc::AddOcl20CompilerOptions(optCompile);
        action.SetOptionList(optCompile);

        const Dataset addedPch;
        action.Do(AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS, inputs, addedPch);
        const Dataset compiledBc;
        action.Do(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, addedPch, compiledBc);

        OptionList optLink;
        optLink.push_back("wavefrontsize64");
        for(const auto& opt : optCompile)
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
        const Dataset addedDevLibs;
        action.Do(AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES, compiledBc, addedDevLibs);
        const Dataset linkedBc;
        action.Do(AMD_COMGR_ACTION_LINK_BC_TO_BC, addedDevLibs, linkedBc);

        action.SetOptionList(optCompile);
        const Dataset relocatable;
        action.Do(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, linkedBc, relocatable);

        action.SetOptionList(OptionList());
        const Dataset exe;
        action.Do(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, relocatable, exe);

        constexpr auto INTENTIONALY_UNKNOWN = static_cast<amd_comgr_status_t>(0xffff);
        if(exe.GetDataCount(AMD_COMGR_DATA_KIND_EXECUTABLE) < 1)
            throw ComgrError{INTENTIONALY_UNKNOWN, "Executable binary not found"};
        // Assume that the first exec data contains the binary we need.
        const auto data = exe.GetData(AMD_COMGR_DATA_KIND_EXECUTABLE, 0);
        data.GetBytes(binary);
    }
    catch(ComgrError& ex)
    {
        MIOPEN_LOG_E("comgr status = " << GetStatusText(ex.status));
        if(!ex.text.empty())
            MIOPEN_LOG_W(ex.text);
    }
}

} // namespace comgr
} // namespace miopen
