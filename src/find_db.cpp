/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/find_db.hpp>

#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/perf_field.hpp>
#if MIOPEN_EMBED_DB
#include <miopen_data.hpp>
#endif
#include <miopen/filesystem.hpp>
#include <string>
#include <vector>

namespace miopen {

namespace debug {

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
MIOPEN_EXPORT bool testing_find_db_enabled = true;

/// \todo Remove when #1723 is resolved.
boost::optional<std::string>& testing_find_db_path_override()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static boost::optional<std::string> data = boost::none;
    return data;
}

} // namespace debug

#if MIOPEN_EMBED_DB
template <class TDb>
std::string FindDbRecord_t<TDb>::GetInstalledPathEmbed(Handle& handle,
                                                       const std::string& path_suffix)
{
    static const auto embed_path = [&] {
        const std::string ext = ".fdb.txt";
        const auto root_path  = fs::path(GetSystemDbPath());
        const auto base_name  = handle.GetDbBasename();
        const auto suffix     = GetSystemFindDbSuffix() + path_suffix;
        const auto filename   = base_name + "." + suffix + ext;
        const auto file_path  = root_path / filename;
        if(miopen_data().find(make_object_file_name(filename).string()) != miopen_data().end())
        {
            MIOPEN_LOG_I2("Found exact embedded find database file:" << filename);
            return file_path.string();
        }
        else
        {
            MIOPEN_LOG_I2("inexact find database search");
            std::vector<fs::path> all_files;
            for(const auto& kinder : miopen_data())
            {
                const auto& fname    = kinder.first.substr(0, kinder.first.size() - 2);
                const auto& filepath = root_path / fname;
                if(EndsWith(fname, path_suffix + ".fdb.txt"))
                    all_files.push_back(filepath);
            }

            const auto db_id        = handle.GetTargetProperties().DbId();
            const int real_cu_count = handle.GetMaxComputeUnits();
            int closest_cu          = std::numeric_limits<int>::max();
            fs::path best_path;
            for(const auto& entry : all_files)
            {
                const auto fname = entry.stem().string();
                MIOPEN_LOG_I2("Checking embedded find db file: " << fname);
                // Check for alternate back end same ASIC
                if(fname.rfind(base_name, 0) == 0)
                {
                    return entry.string();
                }
                if(db_id.empty() || !miopen::StartsWith(db_id, "gfx") || real_cu_count == 0)
                    return std::string();
                // Check for alternate ASIC any back end
                if(fname.rfind(db_id, 0) == 0)
                {
                    const auto pos = fname.find('_');
                    int cur_count  = -1;
                    if(pos != std::string::npos)
                        cur_count = std::stoi(fname.substr(pos + 1));
                    else
                        cur_count = std::stoi(fname.substr(db_id.length()), nullptr, 16);
                    if(abs(cur_count - real_cu_count) < (closest_cu))
                    {
                        best_path  = entry;
                        closest_cu = abs(cur_count - real_cu_count);
                    }
                }
            }
            return best_path.string();
        }
    }();
    return embed_path;
}

#else

template <class TDb>
std::string FindDbRecord_t<TDb>::GetInstalledPathFile(Handle& handle,
                                                      const std::string& path_suffix)
{
    static const auto installed_path = [&] {
        const std::string ext = ".fdb.txt";
        const auto root_path  = fs::path(GetSystemDbPath());
        const auto base_name  = handle.GetDbBasename();
        const auto suffix =
            GetSystemFindDbSuffix() + (path_suffix.empty() ? "" : ('.' + path_suffix));
        const auto file_path = root_path / (base_name + "." + suffix + ext);
        if(fs::exists(file_path))
        {
            MIOPEN_LOG_I2("Found exact find database file: " << file_path);
            return file_path.string();
        }
        else
        {
            MIOPEN_LOG_I2("inexact find database search");
            if(fs::exists(root_path) && fs::is_directory(root_path))
            {
                MIOPEN_LOG_I2("Iterating over find db directory " << root_path);
                std::vector<fs::path> all_files;
                std::vector<fs::path> contents;
                std::copy(fs::directory_iterator(root_path),
                          fs::directory_iterator(),
                          std::back_inserter(contents));
                for(auto const& filepath : contents)
                {
                    if(fs::is_regular_file(filepath) &&
                       filepath.extension() == path_suffix + ".fdb.txt")
                        all_files.push_back(filepath);
                }

                const auto db_id        = handle.GetTargetProperties().DbId();
                const int real_cu_count = handle.GetMaxComputeUnits();
                int closest_cu          = std::numeric_limits<int>::max();
                fs::path best_path;
                for(const auto& entry : all_files)
                {
                    const auto fname = entry.stem().string();
                    MIOPEN_LOG_I("Checking find db file: " << fname);
                    // Check for alternate back end same ASIC
                    if(fname.rfind(base_name, 0) == 0)
                    {
                        return entry.string();
                    }
                    if(db_id.empty() || !miopen::StartsWith(db_id, "gfx") || real_cu_count == 0)
                        return std::string();
                    // Check for alternate ASIC any back end
                    if(fname.rfind(db_id, 0) == 0)
                    {
                        const auto pos = fname.find('_');
                        int cur_count  = -1;
                        if(pos != std::string::npos)
                            cur_count = std::stoi(fname.substr(pos + 1));
                        else
                            cur_count = std::stoi(fname.substr(db_id.length()), nullptr, 16);
                        if(abs(cur_count - real_cu_count) < (closest_cu))
                        {
                            best_path  = entry;
                            closest_cu = abs(cur_count - real_cu_count);
                        }
                    }
                }
                return best_path.string();
            }
            else
            {
                MIOPEN_LOG_I("Database directory does not exist");
                return std::string();
            }
        }
    }();
    return installed_path;
}
#endif
template <class TDb>
std::string FindDbRecord_t<TDb>::GetInstalledPath(Handle& handle, const std::string& path_suffix)
{
#if !MIOPEN_DISABLE_SYSDB
#if MIOPEN_EMBED_DB
    return GetInstalledPathEmbed(handle, path_suffix);
#else
    return GetInstalledPathFile(handle, path_suffix);
#endif
#else
    std::ignore = handle;
    std::ignore = path_suffix;
    return "";
#endif
}

template <class TDb>
std::string FindDbRecord_t<TDb>::GetUserPath(Handle& handle, const std::string& path_suffix)
{
#if !MIOPEN_DISABLE_USERDB
    std::ostringstream ss;
    ss << GetUserDbPath().string() << '/';
    ss << handle.GetDbBasename();
    ss << '.' << GetUserDbSuffix();
    if(!path_suffix.empty())
        ss << '.' << path_suffix;
    ss << ".ufdb.txt";
    return ss.str();
#else
    std::ignore = handle;
    std::ignore = path_suffix;
    return "";
#endif
}

template <class TDb>
bool FindDbRecord_t<TDb>::Validate(Handle& handle, const NetworkConfig& config) const
{
    auto unbuilt = false;
    auto any     = false;

    for(const auto& pair : content->As<FindDbData>())
    {
        if(in_sync)
        {
            if(!handle.GetInvoker(config, {{pair.first}}))
            {
                unbuilt = true;
                // This is not an logged as error because no error was detected.
                // Find wasn't executed yet and invokers were not prepared.
                LogFindDbItem(pair);
                break;
            }

            any = true;
            continue;
        }
    }

    return !any || unbuilt;
}

template <class TDb>
void FindDbRecord_t<TDb>::CopyTo(std::vector<PerfField>& to) const
{
    const auto range = content->As<FindDbData>();
    std::transform(range.begin(), range.end(), std::back_inserter(to), [](const auto& pair) {
        return PerfField{
            pair.second.algorithm, pair.first, pair.second.time, pair.second.workspace};
    });
}

template <class TDb>
void FindDbRecord_t<TDb>::LogFindDbItem(const std::pair<std::string, FindDbData>& item) const
{
    MIOPEN_LOG_I2("Kernel cache entry not found for solver: "
                  << item.first << " at network config: " << content->GetKey());

    for(const auto& pair2 : content->As<FindDbData>())
        MIOPEN_LOG_I2("Find-db record content: " << pair2.first << ':' << pair2.second);
}

template class FindDbRecord_t<FindDb>;
template class FindDbRecord_t<UserFindDb>;

} // namespace miopen
