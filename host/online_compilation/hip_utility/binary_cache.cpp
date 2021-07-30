/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <binary_cache.hpp>
#include <handle.hpp>
#include <md5.hpp>
#include <env.hpp>
#include <stringutils.hpp>
#include <logger.hpp>
#include <target_properties.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

namespace olCompile {

OLC_DECLARE_ENV_VAR(OLC_DISABLE_CACHE)
OLC_DECLARE_ENV_VAR(HOME)

static boost::filesystem::path ComputeCachePath()
{
    const char* home_dir = GetStringEnv(HOME{});
    if(home_dir == nullptr || home_dir == std::string("/") || home_dir == std::string(""))
    {
        home_dir = "/tmp";
    }

    auto p = boost::filesystem::path{home_dir} / "_hip_binary_kernels_";

    if(!boost::filesystem::exists(p))
        boost::filesystem::create_directories(p);
    return p;
}

boost::filesystem::path GetCachePath()
{
    static const boost::filesystem::path user_path = ComputeCachePath();

    return user_path;
}

static bool IsCacheDisabled() { return olCompile::IsEnabled(OLC_DISABLE_CACHE{}); }

boost::filesystem::path
GetCacheFile(const std::string& device, const std::string& name, const std::string& args)
{
    // std::string filename = (is_kernel_str ? olCompile::md5(name) : name) + ".o";
    std::string filename = name + ".o";
    return GetCachePath() / olCompile::md5(device + ":" + args) / filename;
}

boost::filesystem::path LoadBinary(const TargetProperties& target,
                                   const size_t num_cu,
                                   const std::string& name,
                                   const std::string& args)
{
    if(olCompile::IsCacheDisabled())
        return {};

    (void)num_cu;
    auto f = GetCacheFile(target.DbId(), name, args);
    if(boost::filesystem::exists(f))
    {
        return f.string();
    }
    else
    {
        return {};
    }
}

void SaveBinary(const boost::filesystem::path& binary_path,
                const TargetProperties& target,
                const std::string& name,
                const std::string& args)
{
    if(olCompile::IsCacheDisabled())
    {
        boost::filesystem::remove(binary_path);
    }
    else
    {
        auto p = GetCacheFile(target.DbId(), name, args);
        boost::filesystem::create_directories(p.parent_path());
        boost::filesystem::rename(binary_path, p);
    }
}

} // namespace olCompile
