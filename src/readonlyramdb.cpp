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

#include <miopen/readonlyramdb.hpp>
#include <miopen/logger.hpp>
#include <miopen/errors.hpp>
#include <miopen/filesystem.hpp>

#if MIOPEN_EMBED_DB
#include <miopen_data.hpp>
#endif

#include <fstream>
#include <mutex>
#include <sstream>
#include <map>

namespace miopen {

namespace debug {
bool& rordb_embed_fs_override()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool data = false;
    return data;
}
} // namespace debug

ReadonlyRamDb&
ReadonlyRamDb::GetCached(DbKinds db_kind_, const fs::path& path, bool warn_if_unreadable)
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock{mutex};

    // We don't have to store kind to properly index as different dbs would have different paths
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static auto instances = std::map<fs::path, std::unique_ptr<ReadonlyRamDb>>{};
    const auto it         = instances.find(path);

    if(it != instances.end())
        return *it->second;

    auto& instance =
        *instances.emplace(path, std::make_unique<ReadonlyRamDb>(db_kind_, path)).first->second;
    instance.Prefetch(warn_if_unreadable);
    return instance;
}

template <class TFunc>
static auto Measure(const std::string& funcName, TFunc&& func)
{
    if(!miopen::IsLogging(LoggingLevel::Info))
        return func();

    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    MIOPEN_LOG_I("ReadonlyRamDb::" << funcName << " time: " << (end - start).count() * .000001f
                                   << " ms");
}

void ReadonlyRamDb::ParseAndLoadDb(std::istream& input_stream, bool warn_if_unreadable)
{
    if(!input_stream)
    {
        const auto log_level = (warn_if_unreadable && !MIOPEN_DISABLE_SYSDB) ? LoggingLevel::Warning
                                                                             : LoggingLevel::Info;
        MIOPEN_LOG(log_level, "File is unreadable: " << db_path);
        return;
    }

    auto line   = std::string{};
    auto n_line = 0;

    while(std::getline(input_stream, line))
    {
        ++n_line;

        if(line.empty())
            continue;

        const auto key_size = line.find('=');
        const bool is_key   = (key_size != std::string::npos && key_size != 0);

        if(!is_key)
        {
            MIOPEN_LOG_E("Ill-formed record: key not found: " << db_path << "#" << n_line);
            continue;
        }

        const auto key      = line.substr(0, key_size);
        const auto contents = line.substr(key_size + 1);

        cache.emplace(key, CacheItem{n_line, contents});
    }
}

void ReadonlyRamDb::Prefetch(bool warn_if_unreadable)
{
    Measure("Prefetch", [this, warn_if_unreadable]() {
        if(db_path.empty())
            return;
        constexpr bool isEmbedded = MIOPEN_EMBED_DB;
        // cppcheck-suppress knownConditionTrueFalse
        if(!debug::rordb_embed_fs_override() && isEmbedded)
        {
#if MIOPEN_EMBED_DB
            fs::path filepath(db_path);
            const auto& it_p = miopen_data().find(make_object_file_name(filepath.filename()));
            if(it_p == miopen_data().end())
                MIOPEN_THROW(miopenStatusInternalError,
                             "Unknown database: " + filepath.filename() +
                                 " in internal filesystem");

            const auto& p = it_p->second;
            ptrdiff_t sz  = p.second - p.first;
            MIOPEN_LOG_I2("Loading In Memory file: " << filepath);
            auto input_stream = std::stringstream(std::string(p.first, sz));
            ParseAndLoadDb(input_stream, warn_if_unreadable);
#endif
        }
        else
        {
            auto input_stream = std::ifstream{db_path};
            ParseAndLoadDb(input_stream, warn_if_unreadable);
        }
    });
}
} // namespace miopen
