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
#include <miopen/sqlite_db.hpp>
#if MIOPEN_EMBED_DB
#include <miopen_data.hpp>
#endif
#include <boost/filesystem.hpp>

#include <string>

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

struct ExecutionContext
{
    // Operation modes & environment
    bool do_search               = false;
    bool save_srch_req           = false;
    bool use_asm_kernels         = false;
    bool use_hip_kernels         = true;
    bool use_opencl_convolutions = true;
    bool use_binaries            = true;
    rocm_meta_version rmv        = rocm_meta_version::Default;
    bool disable_search_enforce  = false;
    // Skip perf-db reads and use the default performance configuration. This is used, for example,
    // to optimize the getWorkspaceSize() calls for speed. This specific optimization is correct
    // because Solvers shall be written so that the required workspace size does not depend on the
    // performance config.
    bool disable_perfdb_access      = false;
    bool use_dynamic_solutions_only = false;

    inline Handle& GetStream() const { return *stream; }
    inline void SetStream(Handle* stream_) { stream = stream_; }

    ExecutionContext() = default;
    ExecutionContext(Handle* stream_) : stream(stream_) {}

    void DetectRocm();
#if MIOPEN_EMBED_DB
    std::string GetPerfDbPathEmbed() const
    {
        static const auto result = [&] {
            boost::filesystem::path pdb_path(GetSystemDbPath());
            std::ostringstream filename;
            // clang-format off
            filename << GetStream().GetDbBasename();
#if MIOPEN_ENABLE_SQLITE
            const std::string ext = ".db";
#else
            const std::string ext = ".cd.pdb.txt";
#endif
            filename << ext;
            // clang-format on
            if(miopen_data().find(filename.str() + ".o") != miopen_data().end())
            {
                MIOPEN_LOG_I("Found exact embedded perf database file");
                return (pdb_path / filename.str()).string();
            }
            else
            {
                MIOPEN_LOG_I2("inexact embedded perf database search");
                const auto db_id        = GetStream().GetTargetProperties().DbId();
                const int real_cu_count = GetStream().GetMaxComputeUnits();
                namespace fs            = boost::filesystem;
                int closest_cu          = std::numeric_limits<int>::max();
                fs::path best_path;
                for(auto const& entry : miopen_data())
                {
                    // string the .o from the filename
                    const auto fname = entry.first.substr(0, entry.first.size() - 2);
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
                            MIOPEN_LOG_I2("Updating best candidate to: " << filepath.string());
                            best_path  = filepath;
                            closest_cu = abs(cur_count - real_cu_count);
                        }
                    }
                }
                return best_path.string();
            }
            return std::string();
        }();
        return result;
    }
#else
    std::string GetPerfDbPathFile() const
    {
        static const auto result = [&] {
            boost::filesystem::path pdb_path(GetSystemDbPath());
            std::ostringstream filename;
            // clang-format off
        filename << GetStream().GetDbBasename();
#if MIOPEN_ENABLE_SQLITE
        const std::string ext = ".db";
#else
        const std::string ext = ".cd.pdb.txt";
#endif
        filename << ext;
            // clang-format on
            if(boost::filesystem::exists(pdb_path / filename.str()))
            {
                MIOPEN_LOG_I("Found exact perf database file");
                return (pdb_path / filename.str()).string();
            }
            else
            {
                MIOPEN_LOG_I2("inexact perf database search");
                const auto db_id        = GetStream().GetTargetProperties().DbId();
                const int real_cu_count = GetStream().GetMaxComputeUnits();
                namespace fs            = boost::filesystem;
                if(fs::exists(pdb_path) && fs::is_directory(pdb_path))
                {
                    MIOPEN_LOG_I2("Iterating over perf db directory " << pdb_path.string());
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
                           fname.rfind(db_id, 0) == 0) // starts with db_id
                        {
                            MIOPEN_LOG_I2("Checking perf db file: " << fname);
                            const auto pos = fname.find('_');
                            int cur_count  = -1;
                            try
                            {
                                if(pos != std::string::npos)
                                    cur_count = std::stoi(fname.substr(pos + 1));
                                else
                                    cur_count =
                                        std::stoi(fname.substr(db_id.length()), nullptr, 16);
                            }
                            catch(const std::exception& e)
                            {
                                MIOPEN_LOG_I2("Unable to infer CU count for file: "
                                              << fname << " : " << e.what());
                                continue;
                            }

                            if(abs(cur_count - real_cu_count) < (closest_cu))
                            {
                                MIOPEN_LOG_I2("Updating best candidate to: " << filepath.string());
                                best_path  = filepath;
                                closest_cu = abs(cur_count - real_cu_count);
                            }
                        }
                    }
                    return best_path.string();
                }
                else
                {
                    MIOPEN_LOG_I("Database directory does not exist");
                }
            }
            return std::string();
        }();
        return result;
    }
#endif

    std::string GetPerfDbPath() const
    {
#if MIOPEN_EMBED_DB
        return GetPerfDbPathEmbed();
#else
        return GetPerfDbPathFile();
#endif
    }

    std::string GetUserPerfDbPath() const
    {
        // an empty user-db path indicates user intent to disable
        // the database. Default in when dev builds are on
        // clang-format off
	const auto& udb = GetUserDbPath();
	if(udb.empty())
		return "";
        boost::filesystem::path pdb_path(udb);
        std::ostringstream filename;
        filename << GetStream().GetDbBasename();
#if MIOPEN_ENABLE_SQLITE
        filename << "_" << SQLitePerfDb::MIOPEN_PERFDB_SCHEMA_VER << ".udb";
#else
        filename << "."
             << GetUserDbSuffix()
             << ".cd.updb.txt";
#endif
        // clang-format on
        return (pdb_path / filename.str()).string();
    }

    private:
    Handle* stream = nullptr;
};

class AutoUseFastDynamicSolutions
{
    bool prev_use_dynamic_;
    ExecutionContext* const ctx;

    public:
    AutoUseFastDynamicSolutions(ExecutionContext& ctx_) : ctx(&ctx_)
    {
        prev_use_dynamic_ = ctx->use_dynamic_solutions_only;

        ctx->use_dynamic_solutions_only = true;
    }

    ~AutoUseFastDynamicSolutions() { ctx->use_dynamic_solutions_only = prev_use_dynamic_; }
};

bool IsHipKernelsEnabled();

} // namespace miopen
