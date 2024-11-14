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

#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include "get_handle.hpp"
#include <miopen/readonlyramdb.hpp>
#include <miopen/execution_context.hpp>

#include <miopen/find_db.hpp>
#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/mt_queue.hpp>
#include <miopen/filesystem.hpp>

#include <cstdlib>
#include <regex>
#include <exception>
#include <unordered_set>

/// \todo HACK
/// This should be set to 1 if either WORKAROUND_ISSUE_2492_GRANULARITY_LOSS
/// or WORKAROUND_ISSUE_2492_TINY_TENSOR is defined as non-zero in
/// src/solver/conv_winoRxS.cpp
#define WORKAROUND_ISSUE_2492 1

#if WORKAROUND_ISSUE_2492 && defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#define WORKAROUND_ISSUE_1987 0      // Allows testing FDB on gfx1030 (legacy fdb).
#define SKIP_KDB_PDB_TESTING 0       // Allows testing FDB on gfx1030.
#define SKIP_CONVOCLDIRECTFWDFUSED 0 // Allows testing FDB on gfx1030 (legacy fdb).

namespace fs  = miopen::fs;
namespace env = miopen::env;

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DBSYNC_CLEAN)

struct KDBKey
{
    fs::path program_file;
    std::string program_args;
    bool operator==(const KDBKey& other) const
    {
        return (program_file == other.program_file) && (program_args == other.program_args);
    }
};

template <>
struct std::hash<KDBKey>
{
    std::size_t operator()(const KDBKey& k) const
    {
        return std::hash<std::string>()(k.program_file.string()) ^
               (hash<string>()(k.program_args) << 1) >> 1;
    }
};

#if WORKAROUND_ISSUE_2492 && !defined(_WIN32)
static void SetEnvironmentVariable(std::string_view name, std::string_view value)
{
    const auto ret = setenv(name.data(), value.data(), 1);
    ASSERT_TRUE(ret == 0);
}
#endif // WORKAROUND_ISSUE_2492

#if WORKAROUND_ISSUE_1987
/// \todo Copied from src/db_record.cpp
/// Transform find-db (v.1.0) ID:VALUES to the current format.
/// Implementation is intentionally straightforward.
/// Do not include the 1st value from VALUES (solver name) into transformed VALUES.
/// Ignore FdbKCache_Key pair (last two values).
/// Append id (algorithm) to VALUES.
/// Use solver name as ID.
static bool TransformFindDbItem10to20(std::string& id, std::string& values)
{
    MIOPEN_LOG_T("Legacy find-db item: " << id << ':' << values);
    std::size_t pos = values.find(',');
    if(pos == std::string::npos)
        return false;
    const auto solver = values.substr(0, pos);

    const auto time_workspace_pos = pos + 1;
    pos                           = values.find(',', time_workspace_pos);
    if(pos == std::string::npos)
        return false;
    pos = values.find(',', pos + 1);
    if(pos == std::string::npos)
        return false;
    const auto time_workspace = values.substr(time_workspace_pos, pos - time_workspace_pos);

    values = time_workspace + ',' + id;
    id     = solver;
    MIOPEN_LOG_T("Transformed find-db item: " << id << ':' << values);
    return true;
}
#endif

namespace miopen {
conv::Direction GetDirectionFromString(const std::string& direction)
{
    if(direction == "F")
        return conv::Direction::Forward;
    else if(direction == "B")
        return conv::Direction::BackwardData;
    else if(direction == "W")
        return conv::Direction::BackwardWeights;
    throw std::runtime_error("Invalid Direction");
}
miopenTensorLayout_t GetLayoutFromString(const std::string& layout)
{
    if(layout == "NCHW")
        return miopenTensorNCHW;
    else if(layout == "NHWC")
        return miopenTensorNHWC;
    else if(layout == "NCDHW")
        return miopenTensorNCDHW;
    else if(layout == "NDHWC")
        return miopenTensorNDHWC;
    throw std::runtime_error("Invalid Layout");
}
miopenDataType_t GetDataTypeFromString(const std::string& data_type)
{
    if(data_type == "FP32")
        return miopenFloat;
    else if(data_type == "FP16")
        return miopenHalf;
    else if(data_type == "INT8")
        return miopenInt8;
    else if(data_type == "INT32")
        return miopenInt32;
    else if(data_type == "BF16")
        return miopenBFloat16;
    else if(data_type == "FP64")
        return miopenDouble;
    throw std::runtime_error("Invalid data type in find db key");
}
void ParseProblemKey(const std::string& key_, conv::ProblemDescription& prob_desc)
{
    std::string key = key_;
    const auto opt  = SplitDelim(key, '_');
    int group_cnt   = 1;
    conv::Direction dir;
    miopenTensorLayout_t in_layout, wei_layout, out_layout;
    size_t out_h, out_w, in_channels, out_channels, in_h, in_w, batchsize, fil_h, fil_w;
    int pad_h, pad_w, conv_stride_h, conv_stride_w, dil_h, dil_w;
    miopenDataType_t precision;
    TensorDescriptor in{};
    TensorDescriptor wei{};
    TensorDescriptor out{};
    ConvolutionDescriptor conv;
    if(opt.size() >= 2)
    {
        key = opt[0];
        ASSERT_TRUE(StartsWith(opt[1], "g"));
        group_cnt = std::stoi(RemovePrefix(opt[1], "g"));
    }
    else
        ASSERT_TRUE(opt.size() == 1); // either there is one optional args or there is none
    // 2d or 3d ?
    const auto is_3d = [&]() {
        const auto pat_3d = std::regex{"[0-9]x[0-9]x[0-9]"};
        return std::regex_search(key, pat_3d);
    }();
    const auto attrs = SplitDelim(key, '-');
    const auto sz    = attrs.size();
    dir              = GetDirectionFromString(attrs[sz - 1]);
    precision        = GetDataTypeFromString(attrs[sz - 2]);
    if(!is_3d)
    {
        ASSERT_TRUE(sz == 15 || sz == 17);
        std::tie(in_layout, wei_layout, out_layout) = [&]() {
            if(sz == 15) // same layout for all tensors
                return std::tuple{GetLayoutFromString(attrs[12]),
                                  GetLayoutFromString(attrs[12]),
                                  GetLayoutFromString(attrs[12])};
            else if(sz == 17)
                return std::tuple{GetLayoutFromString(attrs[12]),
                                  GetLayoutFromString(attrs[13]),
                                  GetLayoutFromString(attrs[14])};
            throw std::runtime_error{"FDB key parsing error"};
        }();

        in_channels             = std::stoi(attrs[0]);
        in_h                    = std::stoi(attrs[1]);
        in_w                    = std::stoi(attrs[2]);
        out_channels            = std::stoi(attrs[4]);
        out_h                   = std::stoi(attrs[5]);
        out_w                   = std::stoi(attrs[6]);
        batchsize               = std::stoi(attrs[7]);
        const auto split_tensor = [](const std::string& s) {
            const auto tmp = miopen::SplitDelim(s, 'x');
            EXPECT_TRUE(tmp.size() == 2) << "Two Dimensional problems need to have two dimensional "
                                            "filters, pads, strides and dilations"; // for 2d keys
            return std::tuple(std::stoi(tmp[0]), std::stoi(tmp[1]));
        };
        std::tie(fil_h, fil_w)                 = split_tensor(attrs[3]);
        std::tie(pad_h, pad_w)                 = split_tensor(attrs[8]);
        std::tie(conv_stride_h, conv_stride_w) = split_tensor(attrs[9]);
        std::tie(dil_h, dil_w)                 = split_tensor(attrs[10]);

        // construct the problem, serialize it and verify the output
        in = TensorDescriptor{precision, in_layout, {batchsize, in_channels, in_h, in_w}};
        if(dir == conv::Direction::Forward)
            wei = TensorDescriptor(
                precision, wei_layout, {out_channels, in_channels / group_cnt, fil_h, fil_w});
        else
            wei = TensorDescriptor(
                precision, wei_layout, {in_channels, out_channels / group_cnt, fil_h, fil_w});
        out = TensorDescriptor{precision, out_layout, {batchsize, out_channels, out_h, out_w}};
        conv =
            ConvolutionDescriptor{{pad_h, pad_w}, {conv_stride_h, conv_stride_w}, {dil_h, dil_w}};
    }
    else
    {
        int pad_d, conv_stride_d, dil_d;
        size_t in_d, out_d, fil_d;
        // 3D case
        ASSERT_TRUE(sz == 17 || sz == 19);
        std::tie(in_layout, wei_layout, out_layout) = [&]() {
            if(sz == 17) // same layout for all tensors
                return std::tuple{GetLayoutFromString(attrs[14]),
                                  GetLayoutFromString(attrs[14]),
                                  GetLayoutFromString(attrs[14])};
            else // if(sz == 19)
                return std::tuple{GetLayoutFromString(attrs[14]),
                                  GetLayoutFromString(attrs[15]),
                                  GetLayoutFromString(attrs[16])};
        }();

        in_channels             = std::stoi(attrs[0]);
        in_d                    = std::stoi(attrs[1]);
        in_h                    = std::stoi(attrs[2]);
        in_w                    = std::stoi(attrs[3]);
        out_channels            = std::stoi(attrs[5]);
        out_d                   = std::stoi(attrs[6]);
        out_h                   = std::stoi(attrs[7]);
        out_w                   = std::stoi(attrs[8]);
        batchsize               = std::stoi(attrs[9]);
        const auto split_tensor = [](const std::string& s) {
            const auto tmp = miopen::SplitDelim(s, 'x');
            EXPECT_TRUE(tmp.size() == 3) << "For a 3D problem, filters, pads, strides and "
                                            "dilations need to be 3D as well"; // for 3d keys
            return std::tuple(std::stoi(tmp[0]), std::stoi(tmp[1]), std::stoi(tmp[2]));
        };
        std::tie(fil_d, fil_h, fil_w)                         = split_tensor(attrs[4]);
        std::tie(pad_d, pad_h, pad_w)                         = split_tensor(attrs[10]);
        std::tie(conv_stride_d, conv_stride_h, conv_stride_w) = split_tensor(attrs[11]);
        std::tie(dil_d, dil_h, dil_w)                         = split_tensor(attrs[12]);

        // construct the problem, serialize it and verify the output
        in = TensorDescriptor{precision, in_layout, {batchsize, in_channels, in_d, in_h, in_w}};
        if(dir == conv::Direction::Forward)
            wei = TensorDescriptor(precision,
                                   wei_layout,
                                   {out_channels, in_channels / group_cnt, fil_d, fil_h, fil_w});
        else
            wei = TensorDescriptor(precision,
                                   wei_layout,
                                   {in_channels, out_channels / group_cnt, fil_d, fil_h, fil_w});
        out =
            TensorDescriptor{precision, out_layout, {batchsize, out_channels, out_d, out_h, out_w}};
        conv = ConvolutionDescriptor{{pad_d, pad_h, pad_w},
                                     {conv_stride_d, conv_stride_h, conv_stride_w},
                                     {dil_d, dil_h, dil_w},
                                     std::vector<int>(3, 0),
                                     1,
                                     1.0};
        conv::ProblemDescription tmp{in, wei, out, conv, dir};
    }
    conv.group_count = group_cnt;
    prob_desc        = conv::ProblemDescription{in, wei, out, conv, dir};
}

struct FDBVal
{
    std::string solver_id;
    std::string vals;
};

void ParseFDBbVal(const std::string& val, std::vector<FDBVal>& fdb_vals)
{
    std::string id_val;
    std::stringstream ss{val};
    while(std::getline(ss, id_val, ';'))
    {
        const auto id_size = id_val.find(':');
        ASSERT_TRUE(id_size != std::string::npos) << "Ill formed value: " << id_val;
        auto id     = id_val.substr(0, id_size);
        auto values = id_val.substr(id_size + 1);
#if WORKAROUND_ISSUE_1987
        /// \todo Copied from src/db_record.cpp
        /// Detect legacy find-db item (v.1.0 ID:VALUES) and transform it to the current format.
        /// For now, *only* legacy find-db record use convolution algorithm as ID, so if ID is
        /// a valid algorithm, then we can safely assume that the item is in legacy format.
        if(IsValidConvolutionDirAlgo(id))
        {
            ASSERT_TRUE(TransformFindDbItem10to20(id, values))
                << "Ill-formed legacy find-db item: " << values;
        }
#endif
        const auto tmp = FDBVal{id, values};
        fdb_vals.emplace_back(tmp);
    }
}

void GetPerfDbVals(const fs::path& filename,
                   const conv::ProblemDescription& problem_config,
                   std::unordered_map<std::string, std::string>& vals,
                   std::string& select_query)
{
#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
    std::string clause;
    std::vector<std::string> values;
    std::tie(clause, values) = problem_config.WhereClause();
    auto sql                 = SQLite{filename.string(), true};
    // clang-format off
        select_query =
            "SELECT solver, params "
            "FROM perf_db "
            "INNER JOIN " + problem_config.table_name() + " "
            "ON perf_db.config = " + problem_config.table_name() +".id "
            "WHERE "
            "( " + clause + " );";
    // clang-format on
    auto stmt = SQLite::Statement{sql, select_query, values};
    while(true)
    {
        auto rc = stmt.Step(sql);
        if(rc == SQLITE_ROW)
            vals.emplace(stmt.ColumnText(0), stmt.ColumnText(1));
        else if(rc == SQLITE_DONE)
            break;
        else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
            throw std::runtime_error(sql.ErrorMessage());
    }
#else
    const auto& perf_db =
        miopen::ReadonlyRamDb::GetCached(miopen::DbKinds::PerfDb, filename.string(), true);
    const auto& perf_db_map = perf_db.GetCacheMap();

    std::ostringstream ss;
    conv::ProblemDescription::VisitAll(problem_config, [&](auto&& value, auto&&) {
        if(ss.tellp() != 0)
            ss << "x";
        ss << value;
    });
    const auto key = ss.str();

    if(perf_db_map.find(key) != perf_db_map.end())
    {
        std::istringstream pdb_line{perf_db_map.at(key).content};
        char fragment[1024];
        while(pdb_line.getline(fragment, 1024, ';'))
        {
            std::string id_val{fragment};
            const auto id_size = id_val.find(':');
            ASSERT_TRUE(id_size != std::string::npos) << "Ill formed value: " << id_val;
            auto id  = id_val.substr(0, id_size);
            auto cfg = id_val.substr(id_size + 1);
            vals.emplace(id, cfg);
        }
        select_query = " Loading " + key + " from " + filename.string();
    }
#endif
}

void RemovePerfDbEntry(const fs::path& filename,
                       const conv::ProblemDescription& problem_config,
                       const std::string& solver)
{
    std::string select_query;
    std::string clause;
    std::vector<std::string> values;
    std::tie(clause, values) = problem_config.WhereClause();
    auto sql                 = SQLite{filename.string(), true};
    // clang-format off
        select_query =
            "DELETE FROM perf_db "
            "WHERE "
            "config in (select id from " + problem_config.table_name() + " "
            "WHERE ( " + clause + " )) AND "
            "solver='" + solver + "';";
    // clang-format on
    auto stmt = SQLite::Statement{sql, select_query, values};
    while(true)
    {
        auto rc = stmt.Step(sql);
        if(rc == SQLITE_DONE)
            break;
        else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
            throw std::runtime_error(sql.ErrorMessage());
    }
}

auto LoadKDBObjects(const fs::path& filename)
{
    std::unordered_set<KDBKey> kdb_cache;
    auto select_query = "SELECT kernel_name, kernel_args from kern_db";
    auto sql          = SQLite{filename.string(), true};
    auto stmt         = SQLite::Statement{sql, select_query};
    int count         = 0;
    std::cout << "Loading kdb entries into cache" << std::endl;
    while(true)
    {
        auto rc = stmt.Step(sql);
        if(rc == SQLITE_ROW)
        {
            ++count;
            const auto kernel_name = stmt.ColumnText(0);
            const auto kernel_args = stmt.ColumnText(1);
            kdb_cache.emplace(KDBKey{kernel_name, kernel_args});
        }
        else if(rc == SQLITE_DONE)
            break;
        else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
            throw std::runtime_error(sql.ErrorMessage());
        if(count % 2000 == 0)
            std::cout << "Loaded " << count << " entries from KDB" << std::endl;
    }

    std::cout << "Done loading " << count << " entries from kdb file: " << filename << std::endl;
    return kdb_cache;
}

bool CheckKDBObjects(const fs::path& filename,
                     const fs::path& kernel_name,
                     const std::string& kernel_args)
{
    static const auto kdb_cache = LoadKDBObjects(filename);
    return kdb_cache.find(KDBKey{kernel_name, kernel_args}) != kdb_cache.end();
}

bool CheckKDBForTargetID(const fs::path& filename)
{
    // clang-format off
    auto select_query = "SELECT count(*) FROM kern_db WHERE ( kernel_args like '-mcpu=%sram-ecc%') OR (kernel_args like '-mcpu=%xnack%')";
    // clang-format on
    auto sql  = SQLite{filename.string(), true};
    auto stmt = SQLite::Statement{sql, select_query};
    int count = 0;
    while(true)
    {
        auto rc = stmt.Step(sql);
        if(rc == SQLITE_ROW)
            count = stmt.ColumnInt64(0);
        else if(rc == SQLITE_DONE)
            break;
        else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
            throw std::runtime_error(sql.ErrorMessage());
    }
    return count != 0;
}

bool CheckKDBJournalMode(const fs::path& filename)
{
    auto journal_query = "PRAGMA journal_mode";
    auto sql           = SQLite{filename.string(), true};
    auto stmt          = SQLite::Statement{sql, journal_query};
    std::string journal_mode;
    while(true)
    {
        auto rc = stmt.Step(sql);
        if(rc == SQLITE_ROW)
            journal_mode = stmt.ColumnText(0);
        else if(rc == SQLITE_DONE)
            break;
        else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
            throw std::runtime_error(sql.ErrorMessage());
    }
    return journal_mode.compare("off") == 0 || journal_mode.compare("delete") == 0;
}

} // namespace miopen

void SetupPaths(fs::path& fdb_file_path,
                fs::path& pdb_file_path,
                fs::path& kdb_file_path,
                const miopen::Handle& handle)
{
    const std::string ext = ".fdb.txt";
    const auto root_path  = miopen::GetSystemDbPath();
    // The base name has to be the test name for each GPU arch we have
    const std::string base_name = handle.GetDbBasename(); // "gfx90a68";
    const std::string suffix    = "HIP";                  // miopen::GetSystemFindDbSuffix();
    fdb_file_path               = root_path / (base_name + "." + suffix + ext);
#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
    pdb_file_path = root_path / (base_name + ".db");
#else
    pdb_file_path = root_path / (base_name + ".db.txt");
#endif
    kdb_file_path = root_path / (handle.GetDeviceName() + ".kdb");
    ASSERT_TRUE(fs::exists(fdb_file_path)) << "Db file does not exist" << fdb_file_path;
    ASSERT_TRUE(fs::exists(pdb_file_path)) << "Db file does not exist" << pdb_file_path;
    ASSERT_TRUE(SKIP_KDB_PDB_TESTING || fs::exists(kdb_file_path))
        << "Db file does not exist" << kdb_file_path;
}

TEST(CPU_DBSync_NONE, KDBTargetID)
{
    fs::path fdb_file_path, pdb_file_path, kdb_file_path;
#if WORKAROUND_ISSUE_2492
    SetEnvironmentVariable("MIOPEN_DEBUG_WORKAROUND_ISSUE_2492", "0");
#endif
    SetupPaths(fdb_file_path, pdb_file_path, kdb_file_path, get_handle());
    std::ignore = fdb_file_path;
    std::ignore = pdb_file_path;
    EXPECT_TRUE(miopen::CheckKDBJournalMode(kdb_file_path));
    EXPECT_FALSE(!SKIP_KDB_PDB_TESTING && miopen::CheckKDBForTargetID(kdb_file_path));
}

bool LogBuildMessage()
{
    MIOPEN_LOG_W("Unable to produce missing binary due to COMGR being enabled");
    return true;
}

void BuildKernel(const fs::path& program_file,
                 const std::string& program_args,
                 [[maybe_unused]] miopen::Handle& handle)
{
    // Build the code object entry
    // This will write the code object in the user kdb which Jenkins can archive
    // This has to be done with the offline clang compiler and not COMGR (or hipRTC) otherwise the
    // code object would be target ID specific
#if MIOPEN_USE_COMGR
    static const bool discard = LogBuildMessage();
    std::ignore               = discard;
    std::ignore               = program_file;
    std::ignore               = program_args;
#else
    try
    {
        auto p = handle.LoadProgram(program_file, program_args, "");
    }
    catch(std::exception&)
    {
        MIOPEN_LOG_W("Exception thrown while building kernel");
    }
#endif
}

using FDBLine = std::pair<std::string, miopen::ReadonlyRamDb::CacheItem>;

void CheckDynamicFDBEntry(size_t thread_index,
                          size_t total_threads,
                          const std::vector<FDBLine>& find_data,
                          const miopen::ExecutionContext& _ctx,
                          std::atomic<size_t>& counter)
{
    fs::path fdb_file_path, pdb_file_path, kdb_file_path;
    auto& handle = _ctx.GetStream();
    SetupPaths(fdb_file_path, pdb_file_path, kdb_file_path, handle);
    std::unordered_set<KDBKey> checked_kdbs;
    // Get list of dynamic solvers
    std::vector<miopen::solver::Id> dyn_solvers;
    for(const auto id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        const auto solv = id.GetSolver();
        if(solv.IsDynamic())
        {
            std::cout << id.ToString() << "Is Dynamic" << std::endl;
            dyn_solvers.push_back(id);
        }
    }
    const auto data_size = find_data.size();
    for(auto kidx = thread_index; kidx < data_size; kidx += total_threads)
    {
        auto ctx           = _ctx;
        const auto& kinder = find_data[kidx];
        miopen::conv::ProblemDescription problem;
        miopen::ParseProblemKey(kinder.first, problem);
        problem.SetupFloats(ctx); // TODO: Check if this is necessary
        std::stringstream ss;
        problem.Serialize(ss);
        ASSERT_TRUE(ss.str() == kinder.first)
            << "Failed to parse FDB key:" << kidx << ":Parsed Key: " << ss.str();
        // Check the kernels for all dynamic solvers exist
        for(const auto& id : dyn_solvers)
        {
            const auto solv = id.GetSolver();
            if(solv.IsApplicable(_ctx, problem))
            {
                auto db                          = miopen::GetDb(_ctx);
                miopen::solver::ConvSolution sol = solv.FindSolution(_ctx, problem, db, {});
                EXPECT_TRUE(sol.Succeeded())
                    << "Applicable solver generated invalid solution fdb-key:" << kinder.first
                    << " Solver: " << id.ToString();
                for(const auto& kern : sol.construction_params)
                {
                    std::string compile_options = kern.comp_options;
                    auto program_file           = miopen::make_object_file_name(kern.kernel_file);
                    ASSERT_TRUE(kern.kernel_file.extension() != ".mlir")
                        << "MLIR detected in dynamic solvers";
                    compile_options += " -mcpu=" + handle.GetDeviceName();
                    auto search = checked_kdbs.find({program_file, compile_options});
                    if(search !=
                       checked_kdbs
                           .end()) // we have reported this object before, no need to check again
                        continue;
                    EXPECT_TRUE(
                        miopen::CheckKDBObjects(kdb_file_path, program_file, compile_options))
                        << "KDB entry not found for fdb-key:" << kinder.first
                        << " Solver: " << id.ToString() << " filename:" << program_file
                        << " compile_args:" << compile_options;
                    checked_kdbs.emplace(KDBKey{program_file, compile_options});
                    BuildKernel(kern.kernel_file, kern.comp_options, handle);
                }
            }
        }
        if(kidx % 100 == 0)
            std::cout << "Lines of find db completed:" << counter << std::endl;
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}

TEST(CPU_DBSync_NONE, DISABLED_DynamicFDBSync)
{
    fs::path fdb_file_path, pdb_file_path, kdb_file_path;
    auto& handle = get_handle();
    SetupPaths(fdb_file_path, pdb_file_path, kdb_file_path, handle);
    miopen::CheckKDBObjects(kdb_file_path, "", "");

    const auto& find_db =
        miopen::ReadonlyRamDb::GetCached(miopen::DbKinds::FindDb, fdb_file_path.string(), true);
    // assert that find_db.cache is not empty, since that indicates the file was not readable
    ASSERT_TRUE(!find_db.GetCacheMap().empty()) << "Find DB does not have any entries";

    auto _ctx = miopen::ExecutionContext{};
    _ctx.SetStream(&handle);

    // Convert the map to a vector
    std::vector<std::pair<std::string, miopen::ReadonlyRamDb::CacheItem>> fdb_data;
    const auto& find_db_map = find_db.GetCacheMap();
    fdb_data.resize(find_db_map.size());
    std::copy(find_db_map.begin(), find_db_map.end(), fdb_data.begin());
    std::atomic<size_t> counter = 0;
    const int total_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), 32);
    std::vector<std::thread> agents;
    agents.reserve(total_threads);
    for(auto idx = 0; idx < total_threads; ++idx)
    {
        agents.emplace_back(CheckDynamicFDBEntry,
                            idx,
                            total_threads,
                            std::cref(fdb_data),
                            std::cref(_ctx),
                            std::ref(counter));
    }

    for(auto idx = 0; idx < total_threads; ++idx)
    {
        agents.at(idx).join();
    }
    ASSERT_TRUE(counter == fdb_data.size())
        << "Multi-threading error, work done is not equal to total work";
}

void CheckFDBEntry(size_t thread_index,
                   size_t total_threads,
                   std::vector<FDBLine>& data,
                   miopen::RamDb& find_db_rw,
                   const miopen::ExecutionContext& _ctx,
                   std::atomic<size_t>& counter)
{
    fs::path fdb_file_path, pdb_file_path, kdb_file_path;
    SetupPaths(fdb_file_path, pdb_file_path, kdb_file_path, _ctx.GetStream());
    std::unordered_set<KDBKey> checked_kdbs;
    const auto data_size = data.size();
    auto failures        = 0;
    for(auto kidx = thread_index; kidx < data_size; kidx += total_threads)
    {
        const auto& kinder = data.at(kidx);
        auto ctx           = _ctx;
        miopen::conv::ProblemDescription problem;
        miopen::ParseProblemKey(kinder.first, problem);
        problem.SetupFloats(ctx); // TODO: Check if this is necessary
        std::stringstream ss;
        problem.Serialize(ss);
        // moment of truth
        EXPECT_TRUE(ss.str() == kinder.first)
            << '[' << (++failures) << "] " //
            << "Failed to parse FDB key:" << kidx << ":Parsed Key: " << ss.str();

        std::vector<miopen::FDBVal> fdb_vals;
        miopen::ParseFDBbVal(kinder.second.content, fdb_vals);

        std::unordered_map<std::string, std::string> pdb_vals;
        std::string pdb_select_query;
        miopen::GetPerfDbVals(pdb_file_path, problem, pdb_vals, pdb_select_query);

        // This is an opportunity to link up fdb and pdb entries
        auto fdb_idx = 0; // check kdb only for the fastest kernel
        for(const auto& val : fdb_vals)
        {
            miopen::solver::Id id{val.solver_id};
            EXPECT_TRUE(id.IsValid())
                << '[' << (++failures) << "] " //
                << "Solver " << id.Value() << "/" << id.ToString() << ", val.solver_id "
                << val.solver_id << ", val.vals " << val.vals;

#if SKIP_CONVOCLDIRECTFWDFUSED
            /// \todo Workaround: solv.IsApplicable() asserts with ConvOclDirectFwdFused
            /// on gfx1030. AnySolver instance is empty (nullptr) due to some unknown reason.
            if(val.solver_id == "ConvOclDirectFwdFused")
            {
                MIOPEN_LOG_I("Skipping: val.solver_id " << val.solver_id << ", val.vals "
                                                        << val.vals);
                ++fdb_idx;
                continue;
            }
#endif

            const auto solv = id.GetSolver();
            // Skip MLIR
            if(miopen::StartsWith(id.ToString(), "ConvMlir"))
            {
                MIOPEN_LOG_I("Skipping MLIR solver");
                ++fdb_idx;
                continue;
            }
            EXPECT_TRUE(solv.IsApplicable(ctx, problem)) //
                << '[' << (++failures) << "] "           //
                << "Solver is not applicable fdb-key:" << kinder.first
                << " Solver: " << id.ToString();
            miopen::solver::ConvSolution sol;

            auto db                     = miopen::GetDb(ctx);
            const auto pdb_entry_exists = pdb_vals.find(val.solver_id) != pdb_vals.end();

            if(solv.IsTunable())
            {
                if(env::enabled(MIOPEN_DBSYNC_CLEAN) && not pdb_entry_exists)
                {
                    MIOPEN_LOG_W("PDB entry does not exist for tunable fdb-key:"
                                 << kinder.first << ": solver" << val.solver_id
                                 << ", Removing entry from fdb");
                    find_db_rw.Remove(kinder.first, id.ToString());
                    MIOPEN_LOG_W("Removal Complete fdb-key:" << kinder.first << ": solver"
                                                             << val.solver_id);
                    continue;
                }
                else
                {
                    EXPECT_TRUE(SKIP_KDB_PDB_TESTING || pdb_entry_exists)
                        << '[' << (++failures) << "] " //
                        << "PDB entry does not exist for tunable fdb-key:" << kinder.first
                        << ": solver" << val.solver_id << " pdb-select-query: " << pdb_select_query;
                }
                std::string perf_cfg = "";
                if(!SKIP_KDB_PDB_TESTING && pdb_entry_exists)
                {
                    perf_cfg = pdb_vals.at(val.solver_id);
                    bool res = solv.TestPerfCfgParams(ctx, problem, perf_cfg);
                    if(env::enabled(MIOPEN_DBSYNC_CLEAN) && not res)
                    {
                        MIOPEN_LOG_W("Invalid perf config found fdb-key:"
                                     << kinder.first << ", Solver" << val.solver_id << ":"
                                     << perf_cfg << ", Removing entry from fdb and pdb");
                        find_db_rw.Remove(kinder.first, id.ToString());
                        db.Remove(problem, id.ToString());
                        MIOPEN_LOG_W("Removal Complete fdb-key:" << kinder.first << ": solver"
                                                                 << val.solver_id);
                        continue;
                    }
                    else
                    {
                        EXPECT_TRUE(res) << '[' << (++failures) << "] " //
                                         << "Invalid perf config found fdb-key:" << kinder.first
                                         << ", Solver: " << val.solver_id << ":" << perf_cfg
                                         << " pdb-select-query: " << pdb_select_query;
                    }
                    // we can verify the pdb entry by passing in an empty string and then comparing
                    // the received solution with the one below or having the find_solution pass out
                    // the serialized string
                    sol = solv.FindSolution(ctx, problem, db, {}, perf_cfg);
                }
                else
                {
                    sol      = solv.FindSolution(ctx, problem, db, {}, "");
                    perf_cfg = " Not Found (Using Default)";
                }
                // TODO Generate the Select query for pdb
                EXPECT_TRUE(sol.Succeeded())
                    << '[' << (++failures) << "] " //
                    << "Invalid solution fdb-key:" << kinder.first << " Solver: " << id.ToString()
                    << " perf config:" << perf_cfg;
                if(!SKIP_KDB_PDB_TESTING && fdb_idx == 0)
                {
                    for(const auto& kern : sol.construction_params)
                    {
                        bool found                  = false;
                        std::string compile_options = kern.comp_options;
                        auto program_file = miopen::make_object_file_name(kern.kernel_file);
                        if(kern.kernel_file.extension() != ".mlir")
                        {
                            auto& handle = ctx.GetStream();
                            compile_options += " -mcpu=" + handle.GetDeviceName();
                        }
                        auto search           = checked_kdbs.find({program_file, compile_options});
                        bool reported_already = search != checked_kdbs.end();
                        if(!reported_already) // we have reported this object before, no need to
                                              // check again
                        {
                            found = miopen::CheckKDBObjects(
                                kdb_file_path, program_file, compile_options);
                            checked_kdbs.emplace(KDBKey{program_file, compile_options});
                        }
                        else
                            found = checked_kdbs.count(KDBKey{program_file, compile_options}) > 0;
                        if(!found)
                            EXPECT_TRUE(found)
                                << '[' << (++failures) << "] " //
                                << "KDB entry not found for  fdb-key:" << kinder.first
                                << ", Solver: " << id.ToString() << " perf config:" << perf_cfg
                                << " filename: " << program_file << " compile args: "
                                << compile_options; // for fdb key, solver id, solver pdb entry and
                                                    // kdb file and args
                        if(!reported_already)
                            BuildKernel(kern.kernel_file, kern.comp_options, ctx.GetStream());
                    }
                }
            }
            else
                EXPECT_TRUE(!pdb_entry_exists)
                    << '[' << (++failures) << "] " //
                    << "Non-Tunable solver found in PDB" << solv.GetSolverDbId();
            ++fdb_idx;
        }
        if(kidx % 100 == 0)
            std::cout << "Lines of find db completed:" << kidx << std::endl;
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}
namespace miopen {
struct TestHandle : Handle
{
    TestHandle(size_t _num_cu) : Handle(), num_cu(_num_cu) {}

// Probably, according to the idea of the author of this test, the number of CUs should have been
// substituted with the value passed to the constructor (which in fact did not happen). After
// https://github.com/ROCm/MIOpen/pull/3175, the method became virtual, the substitution actually
// happened, and the test broke. I disabled that part (since it doesn't work as intended anyway) to
// keep its behavior the same.
#if 1
    std::size_t GetMaxComputeUnits() const override
    {
        if(num_cu == 0)
            return Handle::GetMaxComputeUnits();
        return num_cu;
    }
#endif

    size_t num_cu = 0;
};
} // namespace miopen

static inline miopen::TestHandle& get_test_handle(size_t num_cu)
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static miopen::TestHandle h{num_cu};
    static const std::thread::id id = std::this_thread::get_id();
    if(std::this_thread::get_id() != id)
    {
        std::cout << "Cannot use handle across multiple threads\n";
        std::abort();
    }
    return h;
}

void StaticFDBSync(const std::string& arch, const size_t num_cu)
{
    fs::path fdb_file_path, pdb_file_path, kdb_file_path;
    auto& handle = get_test_handle(num_cu);
    if(handle.GetDeviceName() != arch)
        GTEST_SKIP();
    handle.num_cu = num_cu;
    SetupPaths(fdb_file_path, pdb_file_path, kdb_file_path, handle);
    std::cout << "Handle CU count: " << handle.GetMaxComputeUnits()
              << " Parameter Value: " << num_cu << std::endl;
    std::cout << "FDB: " << fdb_file_path << ", PDB: " << pdb_file_path
              << ", KDB: " << kdb_file_path << std::endl;
#if !SKIP_KDB_PDB_TESTING
    // Warmup the kdb cache
    miopen::CheckKDBObjects(kdb_file_path, "", "");
#endif
    const auto& find_db =
        miopen::ReadonlyRamDb::GetCached(miopen::DbKinds::FindDb, fdb_file_path.string(), true);
    auto& find_db_rw =
        miopen::RamDb::GetCached(miopen::DbKinds::FindDb, fdb_file_path.string(), false);
    // assert that find_db.cache is not empty, since that indicates the file was not readable
    ASSERT_TRUE(!find_db.GetCacheMap().empty()) << "Find DB does not have any entries";

    // Convert the map to a vector
    std::vector<FDBLine> fdb_data;
    const auto& find_db_map = find_db.GetCacheMap();
    fdb_data.resize(find_db_map.size());
    std::copy(find_db_map.begin(), find_db_map.end(), fdb_data.begin());

    auto _ctx = miopen::ExecutionContext{};
    _ctx.SetStream(&handle);

    std::atomic<size_t> counter = 0;
    const int total_threads =
        std::min(std::thread::hardware_concurrency(), static_cast<unsigned int>(32));
    std::vector<std::thread> agents;
    agents.reserve(total_threads);
    for(auto idx = 0; idx < total_threads; ++idx)
        agents.emplace_back(CheckFDBEntry,
                            idx,
                            total_threads,
                            std::ref(fdb_data),
                            std::ref(find_db_rw),
                            std::ref(_ctx),
                            std::ref(counter));

    for(auto idx = 0; idx < total_threads; ++idx)
        agents.at(idx).join();
    EXPECT_TRUE(counter == fdb_data.size())
        << "Multi-threading error, work done is not equal to total work" << counter << " : "
        << fdb_data.size();
}

struct CPU_DBSync_NONE : testing::TestWithParam<std::pair<std::string, size_t>>
{
};

TEST_P(CPU_DBSync_NONE, StaticFDBSync)
{
    std::string arch;
    size_t num_cu;
    std::tie(arch, num_cu) = GetParam();
    StaticFDBSync(arch, num_cu);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_DBSync_NONE,
                         testing::Values(std::make_pair("gfx908", 120),
                                         std::make_pair("gfx90a", 104),
                                         std::make_pair("gfx90a", 110),
                                         std::make_pair("gfx942", 304),
                                         std::make_pair("gfx1030", 36)));
