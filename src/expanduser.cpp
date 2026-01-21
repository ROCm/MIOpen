/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/expanduser.hpp>
#include <miopen/filesystem.hpp>

#include <string>
#ifdef _WIN32
#include <optional>
#include <boost/algorithm/string/replace.hpp>
#endif

#ifdef __linux__
#include <miopen/stringutils.hpp>
#endif

MIOPEN_DECLARE_ENV_VAR_STR(HOME)

#ifdef _WIN32
MIOPEN_DECLARE_ENV_VAR_STR(USERPROFILE, miopen::fs::temp_directory_path().string())
MIOPEN_DECLARE_ENV_VAR_STR(HOMEPATH, miopen::fs::temp_directory_path().string())
MIOPEN_DECLARE_ENV_VAR_STR(HOMEDRIVE)
#endif

namespace miopen {

#ifdef __linux__

namespace {
std::string GetHomeDir()
{
    const auto p = env::value(HOME);
    if(!(p.empty() || p == std::string("/")))
    {
        return p;
    }
    // todo:
    // need to figure out what is the correct thing to do here
    // in tensoflow unit tests run via bazel, $HOME is not set, so this can happen
    // setting home_dir to the /tmp for now
    return fs::temp_directory_path().string();
}
} // namespace

fs::path ExpandUser(const fs::path& path)
{
    static const auto home_dir = GetHomeDir();
    return {ReplaceString(path.string(), "~", home_dir)};
}

#else

namespace {
std::optional<std::pair<std::string::size_type, std::string>> ReplaceVariable(
    std::string_view path, const env::detail::EnvVar<std::string>& t, std::size_t offset = 0)
{
    std::vector<std::string> variables{"$" + std::string{t.name()},
                                       "$env:" + std::string{t.name()},
                                       "%" + std::string{t.name()} + "%"};
    for(auto& variable : variables)
    {
        auto pos{path.find(variable, offset)};
        if(pos != std::string::npos)
        {
            std::string result{path};
            result.replace(pos, variable.length(), t.value<std::string>());
            return {{pos, result}};
        }
    }
    return std::nullopt;
}
} // namespace

fs::path ExpandUser(const fs::path& path)
{
    auto result{ReplaceVariable(path.string(), USERPROFILE)};
    if(!result)
    {
        result = ReplaceVariable(path.string(), HOME);
        if(!result)
        {
            result = ReplaceVariable(path.string(), HOMEDRIVE);
            if(result)
            {
                result = ReplaceVariable(result->second, HOMEPATH, result->first);
                // TODO: if (not result): log warning message that
                //       HOMEDRIVE and HOMEPATH work in conjunction, respectively.
            }
        }
    }
    return {!result ? path : std::get<1>(*result)};
}

#endif

} // namespace miopen
