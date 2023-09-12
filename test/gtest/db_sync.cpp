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
#include <miopen/conv/context.hpp>

#include <miopen/find_db.hpp>
#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/mt_queue.hpp>

#include <regex>
#include <exception>

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
    else if(data_type == "INT8x4")
        return miopenInt8x4;
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
        in  = TensorDescriptor{precision, in_layout, {batchsize, in_channels, in_h, in_w}};
        wei = TensorDescriptor(precision, wei_layout, {out_channels, in_channels, fil_h, fil_w});
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
        in  = TensorDescriptor{precision, in_layout, {batchsize, in_channels, in_d, in_h, in_w}};
        wei = TensorDescriptor(
            precision, wei_layout, {out_channels, in_channels, fil_d, fil_h, fil_w});
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
        auto id        = id_val.substr(0, id_size);
        auto values    = id_val.substr(id_size + 1);
        const auto tmp = FDBVal{id, values};
        fdb_vals.emplace_back(tmp);
    }
}

void GetPerfDbVals(const boost::filesystem::path& filename,
                   const conv::ProblemDescription& problem_config,
                   std::unordered_map<std::string, std::string>& vals,
                   std::string& select_query)
{
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
}

void CheckKDBObjects(const boost::filesystem::path& filename,
                     const std::string& kernel_name,
                     const std::string& kernel_args,
                     bool& found)
{
    // clang-format off
        auto select_query = "SELECT count(*) FROM kern_db WHERE (kernel_name = ?) AND ( kernel_args = ?)";
    // clang-format on 
    const std::vector<std::string> value = {kernel_name, kernel_args};
    auto sql = SQLite{filename.string(), true};
    auto stmt = SQLite::Statement{sql, select_query, value};
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
    found = count != 0;
}

bool CheckKDBForTargetID(const boost::filesystem::path& filename)
{
    // clang-format off
        auto select_query = "SELECT count(*) FROM kern_db WHERE ( kernel_args like '-mcpu=%sram-ecc%') OR (kernel_args like '-mcpu=%xnack%')";
    // clang-format on 
    auto sql = SQLite{filename.string(), true};
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
} // namespace miopen

struct KDBKey
{
    std::string program_file;
    std::string program_args;
    bool operator==(const KDBKey& other) const
    {
        return (program_file == other.program_file) && (program_args == other.program_args);
    }

};

template<>
struct std::hash<KDBKey>
{
    std::size_t operator()(const KDBKey& k) const
    {
        using std::hash;
        using std::string;

        return ((hash<string>()(k.program_file))
                ^ (hash<string>()(k.program_args) << 1) >> 1);
    }
};

void SetupPaths(boost::filesystem::path& fdb_file_path, boost::filesystem::path& pdb_file_path, boost::filesystem::path& kdb_file_path)
{
    auto& handle = get_handle();
    const std::string ext = ".fdb.txt";
    const auto root_path  = boost::filesystem::path(miopen::GetSystemDbPath());
    // The base name has to be the test name for each GPU arch we have
    const std::string base_name =  handle.GetDbBasename(); // "gfx90a68";
    const std::string suffix    = "HIP";      // miopen::GetSystemFindDbSuffix();
    fdb_file_path    = root_path / (base_name + "." + suffix + ext);
    pdb_file_path    = root_path / (base_name + ".db");
    kdb_file_path = root_path / (handle.GetDeviceName() + ".kdb");
    ASSERT_TRUE(boost::filesystem::exists(fdb_file_path)) << "Db file does not exist" << fdb_file_path;
    ASSERT_TRUE(boost::filesystem::exists(pdb_file_path)) << "Db file does not exist" << pdb_file_path;
    ASSERT_TRUE(boost::filesystem::exists(kdb_file_path)) << "Db file does not exist" << kdb_file_path;
}

TEST(DBSync, DISABLED_KDBTargetID)
{
    boost::filesystem::path fdb_file_path, pdb_file_path, kdb_file_path;
    SetupPaths(fdb_file_path, pdb_file_path, kdb_file_path);
    std::ignore = fdb_file_path;
    std::ignore = pdb_file_path;
    EXPECT_FALSE(miopen::CheckKDBForTargetID(kdb_file_path));
}

void BuildKernel(const std::string& program_file, const std::string& program_args)
{
    // Build the code object entry
    // This will write the code object in the user kdb which Jenkins can archive
    // This has to be done with the offline clang compiler and not COMGR (or hipRTC) otherwise the code object would be target ID specific
#if MIOPEN_USE_COMGR
    MIOPEN_LOG_W("Unable to produce missing binary due to COMGR being enabled");
    std::ignore = program_file;
    std::ignore = program_args;
#else
    auto& handle =  get_handle();
    try
    {
        auto p = handle.LoadProgram(program_file, program_args, false, "");
    }
    catch(std::exception& )
    {
        MIOPEN_LOG_W("Exception thrown while building kernel");
    }
#endif
}

TEST(DBSync, DISABLED_DynamicFDBSync)
{
    boost::filesystem::path fdb_file_path, pdb_file_path, kdb_file_path;
    SetupPaths(fdb_file_path, pdb_file_path, kdb_file_path);

    const auto& find_db = miopen::ReadonlyRamDb::GetCached(fdb_file_path.string(), true);
    size_t idx         = 0;
    // assert that find_db.cache is not empty, since that indicates the file was not readable
    ASSERT_TRUE(!find_db.GetCacheMap().empty()) << "Find DB does not have any entries";

     //Get list of dynamic solvers
    std::vector<miopen::solver::Id> dyn_solvers;
    for(const auto id : miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        const auto solv = id.GetSolver();
        if(solv.IsDynamic())
        {
            std::cout << id.ToString() << "Is Dynamic" << std::endl;
            dyn_solvers.push_back(id);
        }
    }

    std::unordered_map<KDBKey, bool> checked_kdbs;
    auto& handle = get_handle();
    auto _ctx     = miopen::ConvolutionContext{};
    _ctx.SetStream(&handle);

    for(const auto& kinder : find_db.GetCacheMap())
    {
        auto ctx = _ctx; 
        miopen::conv::ProblemDescription problem;
        miopen::ParseProblemKey(kinder.first, problem);
        problem.SetupFloats(ctx); // TODO: Check if this is necessary
        std::stringstream ss;
        problem.Serialize(ss);
        // moment of truth
        ++idx; 
        ASSERT_TRUE(ss.str() == kinder.first) << "Failed to parse FDB key:" << idx << ":Parsed Key: " << ss.str();
        // Check the kernels for all dynamic solvers exist
        for(const auto& id : dyn_solvers)
        {
            const auto solv = id.GetSolver();
            if(solv.IsApplicable(_ctx, problem))
            {
                auto  db = miopen::GetDb(_ctx);
                miopen::solver::ConvSolution sol = solv.FindSolution(_ctx, problem, db, {});
                EXPECT_TRUE(sol.Succeeded()) << "Applicable solver generated invalid solution fdb-key:" << kinder.first << " Solver: " << id.ToString();
                for(const auto& kern : sol.construction_params)
                {
                    bool found = false;
                    std::string compile_options = kern.comp_options;
                    std::string program_file = kern.kernel_file + ".o";
                    ASSERT_TRUE(!miopen::EndsWith(kern.kernel_file, ".mlir")) << "MLIR detected in dynamic solvers";
                    compile_options += " -mcpu=" + handle.GetDeviceName();
                    auto search = checked_kdbs.find({program_file, compile_options});
                    if(search != checked_kdbs.end()) // we have reported this object before, no need to check again
                        continue; 
                    miopen::CheckKDBObjects(kdb_file_path, program_file, compile_options, found);
                    EXPECT_TRUE(found) << "KDB entry not found for fdb-key:" << kinder.first << " Solver: " << id.ToString() << " filename:" << program_file << " compile_args:" << compile_options;
                    checked_kdbs.emplace(std::make_pair(KDBKey{program_file, compile_options}, found));
                    BuildKernel(kern.kernel_file, kern.comp_options);
                }
            }
        }
    }
}

TEST(DbSync, DISABLED_StaticFDBSync)
{
    boost::filesystem::path fdb_file_path, pdb_file_path, kdb_file_path;
    SetupPaths(fdb_file_path, pdb_file_path, kdb_file_path);

    const auto& find_db = miopen::ReadonlyRamDb::GetCached(fdb_file_path.string(), true);
    size_t idx         = 0;
    // assert that find_db.cache is not empty, since that indicates the file was not readable
    ASSERT_TRUE(!find_db.GetCacheMap().empty()) << "Find DB does not have any entries";

    std::unordered_map<KDBKey, bool> checked_kdbs;

    auto& handle = get_handle();
    auto _ctx     = miopen::ConvolutionContext{};
    _ctx.SetStream(&handle);
    size_t cnt_finddb_entry = 0;
    for(const auto& kinder : find_db.GetCacheMap())
    {
        auto ctx = _ctx; 
        miopen::conv::ProblemDescription problem;
        miopen::ParseProblemKey(kinder.first, problem);
        problem.SetupFloats(ctx); // TODO: Check if this is necessary
        std::stringstream ss;
        problem.Serialize(ss);
        // moment of truth
        ++idx; 
        EXPECT_TRUE(ss.str() == kinder.first) << "Failed to parse FDB key:" << idx << ":Parsed Key: " << ss.str();

        std::vector<miopen::FDBVal> fdb_vals;
        std::unordered_map<std::string, std::string> pdb_vals;
        miopen::ParseFDBbVal(kinder.second.content, fdb_vals);
        std::string pdb_select_query;
        miopen::GetPerfDbVals(pdb_file_path, problem, pdb_vals, pdb_select_query);
        // This is an opportunity to link up fdb and pdb entries
        auto fdb_idx = 0; // check kdb only for the fastest kernel
        for(const auto& val : fdb_vals)
        {
            miopen::solver::Id id{val.solver_id};
            const auto solv = id.GetSolver();
            // Skip MLIR
            if(miopen::StartsWith(id.ToString(), "ConvMlir"))
            {
                MIOPEN_LOG_I("Skipping MLIR solver");
                ++fdb_idx;
                continue;
            }
            EXPECT_TRUE(solv.IsApplicable(ctx, problem)) << "Solver is not applicable fdb-key:" << kinder.first << " Solver: " << id.ToString();
            miopen::solver::ConvSolution sol;
            if(solv.IsTunable())
            {
                const auto pdb_entry_exists = pdb_vals.find(val.solver_id) != pdb_vals.end();
                // TODO: Print the SQL query
                EXPECT_TRUE(pdb_entry_exists) << "PDB entry does not exist for tunable fdb-key:" << kinder.first << ": solver" << val.solver_id << " pdb-select-query: " << pdb_select_query;
                auto db        = miopen::GetDb(ctx);
                std::string pdb_entry = "";
                if(pdb_entry_exists)
                {
                    pdb_entry = pdb_vals.at(val.solver_id);
                    bool res = solv.TestPerfCfgParams(ctx, problem, pdb_vals.at(val.solver_id));
                    EXPECT_TRUE(res) << "Invalid perf config found fdb-key:" << kinder.first << " Solver: " << solv.GetSolverDbId() << ":" << pdb_vals.at(val.solver_id) << " pdb-select-query: " << pdb_select_query;
                    // we can verify the pdb entry by passing in an empty string and then comparing the received solution with the one below or having the find_solution pass out the serialized string
                    sol = solv.FindSolution(ctx, problem, db, {}, pdb_vals.at(val.solver_id));
                }
                else
                {
                    sol = solv.FindSolution(ctx, problem, db, {}, "");
                    pdb_entry = " Not Found (Using Default)";
                }
                // TODO Generate the Select query for pdb
                EXPECT_TRUE(sol.Succeeded()) << "Invalid solution fdb-key:" << kinder.first << " Solver: " << id.ToString() << " pdb-val:" << pdb_entry;
                if(fdb_idx == 0)
                {
                    for(const auto& kern : sol.construction_params)
                    {
                        bool found = false;
                        std::string compile_options = kern.comp_options;
                        std::string program_file = kern.kernel_file + ".o";
                        if(!miopen::EndsWith(kern.kernel_file, ".mlir"))
                        {
                            compile_options += " -mcpu=" + handle.GetDeviceName();
                        }
                        auto search = checked_kdbs.find({program_file, compile_options});
                        bool reported_already = search != checked_kdbs.end();
                        if(!reported_already) // we have reported this object before, no need to check again
                        {
                            miopen::CheckKDBObjects(kdb_file_path, program_file, compile_options, found);
                            checked_kdbs.emplace(std::make_pair(KDBKey{program_file, compile_options}, found));
                        }
                        else
                            found = checked_kdbs.at(KDBKey{program_file, compile_options});
                        if(!found)
                            EXPECT_TRUE(found) << "KDB entry not found for  fdb-key:" << kinder.first << " Solver: " << id.ToString() << " pdb-val:" << pdb_entry << " filename: " << program_file << " compile args: " << compile_options;// for fdb key, solver id, solver pdb entry and kdb file and args 
                        if(!reported_already)
                            BuildKernel(kern.kernel_file, kern.comp_options);
                    }
                }
            }
            else
                EXPECT_TRUE(pdb_vals.find(val.solver_id) ==
                       pdb_vals.end())  << "Non-Tunable solver found in PDB" << solv.GetSolverDbId() ;
            ++fdb_idx;
        }

        std::cout << "Lines of find db completed:" << ++cnt_finddb_entry << std::endl;
    }

    // for each key , val pair in the find db
    // for each solver entry in the val of find db
    // check that the solver is applicable
    // check the workspace value is correct
    // Make sure the perf db entry is valid
    // make sure the kdb entry exists and is loadable in hip
    // Bonus: No phantom entries in perf db, no phantom entries in kdb

    // Lets look at the dynamic kernels
    // construct the list of dynamic solvers

    // for each entry in the find db, for each applicable dynamic solver make sure we have the required kdb entries 

}
