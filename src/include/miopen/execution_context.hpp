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

#pragma once

#include <miopen/db_path.hpp>
#include <miopen/handle.hpp>
#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif
#if MIOPEN_EMBED_DB
#include <miopen_data.hpp>
#endif
#include <miopen/filesystem.hpp>

#include <string>
#include <string_view>

class rocm_meta_version
{
    int val = Unknown;

public:
    static constexpr int Unknown = 0, // Unset env.vars read as 0.
        AMDHSA_COv2              = 1, // V2 metadata, https://llvm.org/docs/AMDGPUUsage.html
        AMDHSA_COv2_COv3         = 2, // E.g. ROCm 2.10 supports both.
        AMDHSA_COv3              = 3, // V3 metadata, https://llvm.org/docs/AMDGPUUsage.html
        Default                  = AMDHSA_COv3; // Used when auto-detection fails.

private:
    static constexpr int End = 4, Begin = Unknown;

public:
    rocm_meta_version(int v) : val(v) {}
    int getValue() const { return val; }
    bool IsValid() const { return Begin <= val && val < End; }
    bool IsUnknown() const { return val == Unknown; }
    bool IsV2() const { return AMDHSA_COv2 <= val && val <= AMDHSA_COv2_COv3; }
    bool IsV2orV3() const { return AMDHSA_COv2 <= val && val <= AMDHSA_COv3; }
    bool IsV3() const { return AMDHSA_COv2_COv3 <= val && val <= AMDHSA_COv3; }
    bool UseV3() const;
};

namespace miopen {

namespace debug {

/// Inform the library that some warm-up (e.g. the one implemented in the driver)
/// is in progress. The library can use this, for example, to disable some
/// workarounds that would affect warm-up otherwise.
/// WARNING: This switch is not intended for use in multi-threaded applications.
MIOPEN_EXPORT extern bool
    IsWarmupOngoing; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

} // namespace debug

struct MIOPEN_INTERNALS_EXPORT ExecutionContext
{
    // Solution-specific
    std::string general_compile_options;

    // Operation modes & environment
    bool do_search               = false;
    bool db_update               = false;
    bool use_asm_kernels         = false;
    bool use_hip_kernels         = true;
    bool use_opencl_convolutions = true;
    rocm_meta_version rmv        = rocm_meta_version::Default;
    bool disable_search_enforce  = false;
    // Skip perf-db reads and use the default performance configuration. This is used, for example,
    // to optimize the getWorkspaceSize() calls for speed. This specific optimization is correct
    // because Solvers shall be written so that the required workspace size does not depend on the
    // performance config.
    bool disable_perfdb_access      = false;
    bool use_dynamic_solutions_only = false;
    bool is_for_generic_search      = false;

    inline const Handle& GetStream() const { return *stream; }
    inline void SetStream(const Handle* stream_) { stream = stream_; }

    ExecutionContext() { DetectRocm(); }
    ExecutionContext(const Handle* stream_) : stream(stream_) { DetectRocm(); }

    virtual ~ExecutionContext()               = default;
    ExecutionContext(const ExecutionContext&) = default;
    ExecutionContext(ExecutionContext&&)      = default;
    ExecutionContext& operator=(const ExecutionContext&) = default;
    ExecutionContext& operator=(ExecutionContext&&) = default;

#if MIOPEN_EMBED_DB
    fs::path GetPerfDbPathEmbed(std::string_view prefix = "") const
    {
        static const auto result = [&] {
            auto pdb_path(GetSystemDbPath());
            std::ostringstream filename;
            if(!prefix.empty())
                filename << prefix << '_';
            // clang-format off
            filename << GetStream().GetDbBasename();
#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
            const std::string ext = ".db";
#else
            const std::string ext = ".db.txt";
#endif
            filename << ext;
            // clang-format on
            if(miopen_data().find(fs::path(filename.str() + ".o")) != miopen_data().end())
            {
                MIOPEN_LOG_I("Found exact embedded perf database file");
                return pdb_path / filename.str();
            }
            else
            {
                MIOPEN_LOG_I2("inexact embedded perf database search");
                const auto db_id        = GetStream().GetTargetProperties().DbId();
                const int real_cu_count = GetStream().GetMaxComputeUnits();
                int closest_cu          = std::numeric_limits<int>::max();
                fs::path best_path;
                for(auto const& entry : miopen_data())
                {
                    // string the .o from the filename
                    const auto fname = entry.first.stem().string();
                    MIOPEN_LOG_I2("Testing embedded file:" << fname);
                    const auto& filepath = pdb_path / fname;
                    if(filepath.extension() == ext &&
                       fname.rfind(db_id, 0) == 0) // starts with db_id
                    {
                        MIOPEN_LOG_I2("Checking embedded perf db file: " << fname);
                        const auto pos = fname.find('_');
                        int cur_count  = -1;
                        try
                        {
                            if(pos != std::string::npos)
                                cur_count = std::stoi(fname.substr(pos + 1));
                            else
                                cur_count = std::stoi(fname.substr(db_id.length()), nullptr, 16);
                        }
                        catch(const std::exception& e)
                        {
                            MIOPEN_LOG_I2("Unable to infer CU count for file: " << fname << " : "
                                                                                << e.what());
                            continue;
                        }

                        if(abs(cur_count - real_cu_count) < (closest_cu))
                        {
                            MIOPEN_LOG_I2("Updating best candidate to: " << filepath);
                            best_path  = filepath;
                            closest_cu = abs(cur_count - real_cu_count);
                        }
                    }
                }
                return best_path;
            }
            return fs::path();
        }();
        return result;
    }
#else
    fs::path GetPerfDbPathFile(std::string_view prefix = "") const
    {
        static const auto result = [&] {
            const auto pdb_path(GetSystemDbPath());
#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
            constexpr std::string_view ext = ".db";
#else
            constexpr std::string_view ext = ".db.txt";
#endif
            std::string filename{prefix};
            if(!prefix.empty())
                filename.append("_");
            filename.append(GetStream().GetDbBasename());
            filename.append(ext);

            // clang-format on
            if(fs::exists(pdb_path / filename))
            {
                MIOPEN_LOG_I("Found exact perf database file");
                return pdb_path / filename;
            }
            else
            {
                MIOPEN_LOG_I2("inexact perf database search");
                const auto db_id        = GetStream().GetTargetProperties().DbId();
                const int real_cu_count = GetStream().GetMaxComputeUnits();
                if(fs::is_directory(pdb_path))
                {
                    MIOPEN_LOG_I2("Iterating over perf db directory " << pdb_path);
                    int closest_cu = std::numeric_limits<int>::max();
                    fs::path best_path;
                    std::vector<fs::path> contents;
                    std::copy(fs::directory_iterator(pdb_path),
                              fs::directory_iterator(),
                              std::back_inserter(contents));
                    for(auto const& filepath : contents)
                    {
                        const auto fname = filepath.stem().string();
                        if(fs::is_regular_file(filepath) && filepath.extension() == ext &&
                           fname.starts_with(db_id))
                        {
                            MIOPEN_LOG_I2("Checking perf db file: " << fname);
                            const auto pos = fname.find('_');
                            int cur_count  = -1;
                            try
                            {
                                if(pos != std::string::npos)
                                {
                                    cur_count = std::stoi(fname.substr(pos + 1));
                                }
                                else
                                {
                                    cur_count =
                                        std::stoi(fname.substr(db_id.length()), nullptr, 16);
                                }
                            }
                            catch(const std::exception& e)
                            {
                                MIOPEN_LOG_I2("Unable to infer CU count for file: "
                                              << fname << " : " << e.what());
                                continue;
                            }

                            if(abs(cur_count - real_cu_count) < (closest_cu))
                            {
                                MIOPEN_LOG_I2("Updating best candidate to: " << filepath);
                                best_path  = filepath;
                                closest_cu = abs(cur_count - real_cu_count);
                            }
                        }
                    }
                    return best_path;
                }
                else
                {
                    MIOPEN_LOG_I("Database directory does not exist");
                }
            }
            return fs::path{};
        }();
        return result;
    }
#endif

    fs::path GetPerfDbPath(std::string_view prefix = "") const
    {
#if MIOPEN_EMBED_DB
        return GetPerfDbPathEmbed(prefix);
#else
        return GetPerfDbPathFile(prefix);
#endif
    }

    fs::path GetUserPerfDbPath(std::string_view prefix = "") const
    {
        // an empty user-db path indicates user intent to disable
        // the database. Default in when dev builds are on
        const auto& udb = GetUserDbPath();
        if(udb.empty())
            return "";
        std::string filename{prefix};
        if(!prefix.empty())
            filename.append("_");
        filename.append(GetStream().GetDbBasename());
#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
        filename.append("_" + std::string{SQLitePerfDb::MIOPEN_PERFDB_SCHEMA_VER} + ".udb");
#else
        filename.append("." + GetUserDbSuffix() + ".udb.txt");
#endif
        return udb / filename;
    }

private:
    const Handle* stream = nullptr;

    void DetectRocm();
};

struct [[deprecated]] ConvolutionContext : ExecutionContext
{
};

MIOPEN_INTERNALS_EXPORT bool IsHipKernelsEnabled();

} // namespace miopen
