/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#ifdef MIOPEN_BUILD_TESTING

#include <charconv>
#include <cstdlib>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include <miopen/convolution.hpp> // MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL
#include <miopen/env.hpp>
#include <miopen/env_debug.hpp>
#include <miopen/errors.hpp>                  // MIOPEN_THROW
#include <miopen/generic_search_controls.hpp> // MIOPEN_DEBUG_TUNING_ITERATIONS_MAX, MIOPEN_COMPILE_PARALLEL_LEVEL

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_FIND_ONLY_SOLVER)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_WORKAROUND_ISSUE_2492)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING_CMD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_ENFORCE)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_MODE)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_USER_DB_PATH)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_LOG_LEVEL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_PERF_VALS)

namespace miopen {
namespace debug {
namespace env {

namespace {

template <class T>
constexpr bool is_type_bool = std::is_same_v<T, bool>;

template <class T>
constexpr bool is_type_int = (std::is_integral_v<T> && !is_type_bool<T>);

template <class T>
constexpr bool is_type_str = (std::is_same_v<T, std::string> ||
                              std::is_same_v<T, std::string_view>);

struct LibEnvVar
{
    template <class T>
    LibEnvVar(const T& var_in) : impl(std::make_shared<LibEnvVarImpl<T>>(var_in)){};

    std::optional<std::string> Get() const { return impl->Get(); }
    void Update(std::string_view value) const { impl->Update(value); }
    void Clear() const { impl->Clear(); }

private:
    struct LibEnvVarBase
    {
        virtual ~LibEnvVarBase(){};
        virtual std::optional<std::string> Get() const    = 0;
        virtual void Update(std::string_view value) const = 0;
        virtual void Clear() const                        = 0;
    };

    template <class T>
    struct LibEnvVarImpl : LibEnvVarBase
    {
        using value_type = typename T::value_type;
        static_assert(is_type_bool<value_type> || is_type_int<value_type> ||
                      is_type_str<value_type>);

        LibEnvVarImpl(const T& var_in) : var(var_in){};

        std::optional<std::string> Get() const override
        {
            if(!var)
                return std::nullopt;

            const auto value = miopen::env::value(var);

            if constexpr(is_type_bool<value_type>)
            {
                return {value ? "1" : "0"};
            }
            else if constexpr(is_type_int<value_type>)
            {
                return {std::to_string(value)};
            }
            else if constexpr(is_type_str<value_type>)
            {
                return {value};
            }
        }

        void Update(std::string_view value) const override
        {
            if constexpr(is_type_bool<value_type>)
            {
                bool bvalue = (value != "0");
                miopen::env::update(var, bvalue);
            }
            else if constexpr(is_type_int<value_type>)
            {
                value_type ivalue;
                const auto res = std::from_chars(value.data(), value.data() + value.size(), ivalue);
                if(res.ec == std::errc::invalid_argument ||
                   res.ec == std::errc::result_out_of_range)
                {
                    MIOPEN_THROW(miopenStatusInvalidValue,
                                 "Invalid value for env variable: " + value);
                }
                miopen::env::update(var, ivalue);
            }
            else if constexpr(is_type_str<value_type>)
            {
                miopen::env::update(var, value);
            }
        }

        void Clear() const override { miopen::env::clear(var); }

    private:
        const T& var;
    };

    const std::shared_ptr<LibEnvVarBase> impl;
};

const LibEnvVar& FindEnvVariable(std::string_view name)
{
    // MT-Unsafe
    static const std::unordered_map<std::string_view, LibEnvVar> env_variables = {
        // clang-format off
        {MIOPEN_COMPILE_PARALLEL_LEVEL.GetName(), MIOPEN_COMPILE_PARALLEL_LEVEL},
        {MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3.GetName(), MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F2X3},
        {MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3.GetName(), MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3},
        {MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3.GetName(), MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F4X3},
        {MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3.GetName(), MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F5X3},
        {MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3.GetName(), MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F6X3},
        {MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4.GetName(), MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X4},
        {MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5.GetName(), MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X5},
        {MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6.GetName(), MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X6},
        {MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS.GetName(), MIOPEN_DEBUG_CHECK_SUB_BUFFER_OOB_MEMORY_ACCESS},
        {MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW.GetName(), MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW},
        {MIOPEN_DEBUG_CONV_DIRECT.GetName(), MIOPEN_DEBUG_CONV_DIRECT},
        {MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1.GetName(), MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD1X1},
        {MIOPEN_DEBUG_CONV_FFT.GetName(), MIOPEN_DEBUG_CONV_FFT},
        {MIOPEN_DEBUG_CONV_GEMM.GetName(), MIOPEN_DEBUG_CONV_GEMM},
        {MIOPEN_DEBUG_CONV_IMPLICIT_GEMM.GetName(), MIOPEN_DEBUG_CONV_IMPLICIT_GEMM},
        {MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1.GetName(), MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1},
        {MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS.GetName(), MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS},
        {MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS.GetName(), MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS},
        {MIOPEN_DEBUG_CONV_WINOGRAD.GetName(), MIOPEN_DEBUG_CONV_WINOGRAD},
        {MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL.GetName(), MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL},
        {MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS.GetName(), MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS},
        {MIOPEN_DEBUG_FIND_ONLY_SOLVER.GetName(), MIOPEN_DEBUG_FIND_ONLY_SOLVER},
        {MIOPEN_DEBUG_TUNING_ITERATIONS_MAX.GetName(), MIOPEN_DEBUG_TUNING_ITERATIONS_MAX},
        {MIOPEN_DEBUG_WORKAROUND_ISSUE_2492.GetName(), MIOPEN_DEBUG_WORKAROUND_ISSUE_2492},
        {MIOPEN_ENABLE_LOGGING_CMD.GetName(), MIOPEN_ENABLE_LOGGING_CMD},
        {MIOPEN_ENABLE_LOGGING_ELAPSED_TIME.GetName(), MIOPEN_ENABLE_LOGGING_ELAPSED_TIME},
        {MIOPEN_FIND_ENFORCE.GetName(), MIOPEN_FIND_ENFORCE},
        {MIOPEN_FIND_MODE.GetName(), MIOPEN_FIND_MODE},
        {MIOPEN_LOG_LEVEL.GetName(), MIOPEN_LOG_LEVEL},
        {MIOPEN_USER_DB_PATH.GetName(), MIOPEN_USER_DB_PATH},
        {MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_PERF_VALS.GetName(), MIOPEN_DEBUG_CONV_DIRECT_ASM_WRW1X1_PERF_VALS}
        // clang-format on
    };

    const auto& v = env_variables.find(name);
    if(v == env_variables.cend())
    {
        MIOPEN_THROW(miopenStatusNotImplemented, "Environment variable not found: " + name);
    }
    return v->second;
}

} // namespace

std::optional<std::string> GetEnvVariable(std::string_view name)
{
    return FindEnvVariable(name).Get();
}

void UpdateEnvVariable(std::string_view name, std::string_view value)
{
    FindEnvVariable(name).Update(value);
}

void ClearEnvVariable(std::string_view name) { FindEnvVariable(name).Clear(); }

} // namespace env
} // namespace debug
} // namespace miopen

#endif
