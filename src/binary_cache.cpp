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

#include <miopen/binary_cache.hpp>
#include <miopen/handle.hpp>
#include <miopen/md5.hpp>
#include <miopen/errors.hpp>
#include <miopen/env.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/expanduser.hpp>
#include <miopen/miopen.h>
#include <miopen/version.h>
#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif
#include <miopen/kern_db.hpp>
#include <miopen/db.hpp>
#include <miopen/db_path.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/filesystem.hpp>
#include <fstream>
#include <iostream>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DISABLE_CACHE)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_CUSTOM_CACHE_DIR)

namespace miopen {

static fs::path ComputeSysCachePath()
{
    const std::string cache_dir = GetSystemDbPath();
    auto p                      = miopen::ExpandUser(cache_dir);
    if(!fs::exists(p))
        return {};
    else
        return p;
}

static fs::path ComputeUserCachePath()
{
#ifdef MIOPEN_CACHE_DIR
    fs::path p;
    /// If MIOPEN_CUSTOM_CACHE_DIR is set in the environment, then
    /// use exactly that path.
    const auto& custom = miopen::GetStringEnv(ENV(MIOPEN_CUSTOM_CACHE_DIR));
    if(!custom.empty())
    {
        p = ExpandUser(custom);
    }
    else
    {
        const std::string cache_dir = MIOPEN_CACHE_DIR;
        const std::string version   = std::to_string(MIOPEN_VERSION_MAJOR)       //
                                    + "." + std::to_string(MIOPEN_VERSION_MINOR) //
                                    + "." + std::to_string(MIOPEN_VERSION_PATCH) //
                                    + "." + MIOPEN_STRINGIZE(MIOPEN_VERSION_TWEAK);
        p = miopen::ExpandUser(cache_dir) / version;
#if !MIOPEN_BUILD_DEV
        /// \ref nfs-detection
        if(IsNetworkedFilesystem(p))
            p = fs::temp_directory_path();
#endif
    }
    if(!fs::exists(p) && !MIOPEN_DISABLE_USERDB)
        fs::create_directories(p);
    return p;
#else
    return {};
#endif
}

fs::path GetCachePath(bool is_system)
{
    static const fs::path user_path = ComputeUserCachePath();
    static const fs::path sys_path  = ComputeSysCachePath();
    if(is_system)
    {
        if(MIOPEN_DISABLE_SYSDB)
            return {};
        else
            return sys_path;
    }
    else
    {
        if(MIOPEN_DISABLE_USERDB)
            return {};
        else
            return user_path;
    }
}

bool IsCacheDisabled()
{
#ifdef MIOPEN_CACHE_DIR
    if(MIOPEN_DISABLE_USERDB && MIOPEN_DISABLE_SYSDB)
        return true;
    else
        return miopen::IsEnabled(ENV(MIOPEN_DISABLE_CACHE));
#else
    return true;
#endif
}

#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
using KDb = DbTimer<MultiFileDb<KernDb, KernDb, false>>;
KDb GetDb(const TargetProperties& target, size_t num_cu)
{
    static const auto user_dir = ComputeUserCachePath();
    static const auto sys_dir  = ComputeSysCachePath();
    fs::path user_path         = user_dir / (Handle::GetDbBasename(target, num_cu) + ".ukdb");
    fs::path sys_path          = sys_dir / (Handle::GetDbBasename(target, num_cu) + ".kdb");
    if(user_dir.empty())
        user_path = user_dir;
    if(!fs::exists(sys_path))
        sys_path = sys_dir / (target.DbId() + ".kdb");
#if !MIOPEN_EMBED_DB
    if(!fs::exists(sys_path))
        sys_path = fs::path{};
#endif
    return {DbKinds::KernelDb, sys_path.string(), user_path.string()};
}
#endif

fs::path GetCacheFile(const std::string& device, const std::string& name, const std::string& args)
{
    const auto filename = make_object_file_name(name);
    return GetCachePath(false) / miopen::md5(device + ":" + args) / filename;
}

#if MIOPEN_ENABLE_SQLITE_KERN_CACHE
std::vector<char> LoadBinary(const TargetProperties& target,
                             const size_t num_cu,
                             const std::string& name,
                             const std::string& args)
{
    if(miopen::IsCacheDisabled())
        return {};

    auto db = GetDb(target, num_cu);

    const auto filename = make_object_file_name(name).string();
    const KernelConfig cfg{filename, args, {}};

    MIOPEN_LOG_I2("Loading binary for: " << filename << "; args: " << args);
    auto record = db.FindRecord(cfg);
    if(record)
    {
        MIOPEN_LOG_I2("Successfully loaded binary for: " << filename << "; args: " << args);
        return *record;
    }
    else
    {
        MIOPEN_LOG_I2("Unable to load binary for: " << filename << "; args: " << args);
        return {};
    }
}

void SaveBinary(const std::vector<char>& hsaco,
                const TargetProperties& target,
                const std::size_t num_cu,
                const std::string& name,
                const std::string& args)
{
    if(miopen::IsCacheDisabled())
        return;

    auto db = GetDb(target, num_cu);

    const auto filename = make_object_file_name(name).string();
    KernelConfig cfg{filename, args, hsaco};

    MIOPEN_LOG_I2("Saving binary for: " << filename << "; args: " << args);
    db.StoreRecord(cfg);
}
#else
fs::path LoadBinary(const TargetProperties& target,
                    const size_t num_cu,
                    const std::string& name,
                    const std::string& args)
{
    if(miopen::IsCacheDisabled())
        return {};

    (void)num_cu;
    auto f = GetCacheFile(target.DbId(), name, args);
    if(fs::exists(f))
    {
        return f;
    }
    else
    {
        return {};
    }
}

void SaveBinary(const fs::path& binary_path,
                const TargetProperties& target,
                const std::string& name,
                const std::string& args)
{
    if(miopen::IsCacheDisabled())
    {
        fs::remove(binary_path);
    }
    else
    {
        auto p = GetCacheFile(target.DbId(), name, args);
        fs::create_directories(p.parent_path());
        fs::rename(binary_path, p);
    }
}
#endif
} // namespace miopen
