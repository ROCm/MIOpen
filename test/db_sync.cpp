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
#include <miopen/miopen.h>
#include "get_handle.hpp"
#include <miopen/readonlyramdb.hpp>
#include <miopen/conv/context.hpp>

#include <miopen/find_db.hpp>
#include <miopen/tensor.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/any_solver.hpp>

#include <regex>

#if 0
TEST(LOG_TEST, AssertLogCmdOutput)
{
    TestLogFun(miopen::debug::LogCmdConvolution, envConv, logConv, true);
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
    assert(false && "Invalid Direction");
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
    assert(false && "Invalid layout");
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
conv::ProblemDescription ParseProblemKey(const std::string& key_)
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
    // for(const auto& kinder : opt)
    //     std::cout << kinder << std::endl;
    if(opt.size() >= 2)
    {
        key = opt[0];
        assert(StartsWith(opt[1], "g"));
        group_cnt = std::stoi(RemovePrefix(opt[1], "g"));
    }
    else
        assert(opt.size() == 1); // either there is one optional args or there is none
    // 2d or 3d ?
    const auto is_3d = [&]() {
        const auto pat_3d = std::regex{"[0-9]x[0-9]x[0-9]"};
        return std::regex_search(key, pat_3d);
    }();
    const auto attrs = SplitDelim(key, '-');
    // for(const auto& kinder : attrs)
    //     std::cout << kinder << std::endl;
    const auto sz = attrs.size();
    dir           = GetDirectionFromString(attrs[sz - 1]);
    precision     = GetDataTypeFromString(attrs[sz - 2]);
    if(!is_3d)
    {
        std::tie(in_layout, wei_layout, out_layout) = [&]() {
            assert(sz == 15 || sz == 17);
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

        // if(dir == conv::Direction::Forward)
        {
            in_channels  = std::stoi(attrs[0]);
            in_h         = std::stoi(attrs[1]);
            in_w         = std::stoi(attrs[2]);
            out_channels = std::stoi(attrs[4]);
            out_h        = std::stoi(attrs[5]);
            out_w        = std::stoi(attrs[6]);
        }
#if 0
        else
        {
            out_channels = std::stoi(attrs[0]);
            out_h        = std::stoi(attrs[1]);
            out_w        = std::stoi(attrs[2]);
            in_h         = std::stoi(attrs[5]);
            in_w         = std::stoi(attrs[6]);
            in_channels  = std::stoi(attrs[4]);
        }
#endif
        batchsize               = std::stoi(attrs[7]);
        const auto split_tensor = [](const std::string& s) {
            const auto tmp = miopen::SplitDelim(s, 'x');
            assert(tmp.size() == 2); // for 2d keys
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
        std::tie(in_layout, wei_layout, out_layout) = [&]() {
            assert(sz == 17 || sz == 19);
            if(sz == 17) // same layout for all tensors
                return std::tuple{GetLayoutFromString(attrs[14]),
                                  GetLayoutFromString(attrs[14]),
                                  GetLayoutFromString(attrs[14])};
            else // if(sz == 19)
                return std::tuple{GetLayoutFromString(attrs[14]),
                                  GetLayoutFromString(attrs[15]),
                                  GetLayoutFromString(attrs[16])};
        }();

        // if(dir == conv::Direction::Forward)
        {
            in_channels  = std::stoi(attrs[0]);
            in_d         = std::stoi(attrs[1]);
            in_h         = std::stoi(attrs[2]);
            in_w         = std::stoi(attrs[3]);
            out_channels = std::stoi(attrs[5]);
            out_d        = std::stoi(attrs[6]);
            out_h        = std::stoi(attrs[7]);
            out_w        = std::stoi(attrs[8]);
        }
#if 0
        else
        {
            out_channels = std::stoi(attrs[0]);
            out_d        = std::stoi(attrs[1]);
            out_h        = std::stoi(attrs[2]);
            out_w        = std::stoi(attrs[3]);
            in_channels  = std::stoi(attrs[5]);
            in_d         = std::stoi(attrs[6]);
            in_h         = std::stoi(attrs[7]);
            in_w         = std::stoi(attrs[8]);
        }
#endif
        batchsize               = std::stoi(attrs[9]);
        const auto split_tensor = [](const std::string& s) {
            const auto tmp = miopen::SplitDelim(s, 'x');
            assert(tmp.size() == 3); // for 3d keys
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
    conv::ProblemDescription res{in, wei, out, conv, dir};
    return res;
}

std::unordered_map<std::string, std::string> ParseFDBbVal(const std::string& val)
{
    std::string id_val;
    std::unordered_map<std::string, std::string> res;
    std::stringstream ss{val};
    while(std::getline(ss, id_val, ';'))
    {
        const auto id_size = id_val.find(':');
        if(id_size == std::string::npos)
            assert(false); // << "Ill formed value: " << id_val;
        auto id     = id_val.substr(0, id_size);
        auto values = id_val.substr(id_size + 1);
        if(res.find(id) != res.end())
            assert(false); // << "Duplicate value for solver ID:" << id;
        res.emplace(id, values);
    }
    return res;
}

std::unordered_map<std::string, std::string>
GetPerfDbVals(const boost::filesystem::path& filename,
              const conv::ProblemDescription& problem_config)
{
    std::string clause;
    std::vector<std::string> values;
    std::unordered_map<std::string, std::string> res;
    std::tie(clause, values) = problem_config.WhereClause();
    auto sql                 = SQLite{filename.string(), true};
    // clang-format off
        auto select_query =
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
            res.emplace(stmt.ColumnText(0), stmt.ColumnText(1));
        else if(rc == SQLITE_DONE)
            break;
        else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
            MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
    }
    return res;
}
} // namespace miopen

int main(int, char*[])
{
    auto& handle = get_handle();
    auto ctx     = miopen::ConvolutionContext{};
    ctx.SetStream(&handle);
    ctx.DetectRocm();

    const std::string ext = ".fdb.txt";
    const auto root_path  = boost::filesystem::path(miopen::GetSystemDbPath());
    // The base name has to be the test name for each GPU arch we have
    const std::string base_name = "gfx90878"; // handle.GetDbBasename();
    const std::string suffix    = "HIP";      // miopen::GetSystemFindDbSuffix();
    const auto fdb_file_path    = root_path / (base_name + "." + suffix + ext);
    const auto pdb_file_path    = root_path / (base_name + ".db");
    if(!boost::filesystem::exists(fdb_file_path))
        return -1;
    const auto find_db = miopen::ReadonlyRamDb::GetCached(fdb_file_path.string(), true);
    size_t idx         = 0;
    // assert that find_db.cache is not empty, since that indicates the file was not readable
    for(const auto& kinder : find_db.cache)
    {
        const auto problem = miopen::ParseProblemKey(kinder.first);
        problem.SetupFloats(ctx); // TODO: Check if this is necessary
        std::stringstream ss;
        problem.Serialize(ss);
        std::cout << ++idx << ":Parsed Key: " << ss.str() << std::endl;
        assert(ss.str() == kinder.first); // moment of truth
        const auto vals     = miopen::ParseFDBbVal(kinder.second.content);
        const auto pdb_vals = miopen::GetPerfDbVals(pdb_file_path, problem);
        // This is an opportunity to link up fdb and pdb entries

        for(const auto& val : vals)
        {
            std::cout << "Entry: " << val.first << " : " << val.second << std::endl;
            miopen::solver::Id id{val.first};
            const auto solv = id.GetSolver();
            if(!solv.IsApplicable(ctx, problem))
                std::cout << "Solver is not applicable" << std::endl;
            // const auto sol = solv.FindSolution(ctx, problem, miopen::GetDb(ctx), {},
            // pdb_vals.at(val.first));
            if(solv.IsTunable())
            {
                assert(pdb_vals.find(val.first) != pdb_vals.end());
                bool res = solv.TestPerfCfgParams(ctx, problem, pdb_vals.at(val.first));
                assert(res);
                auto db        = miopen::GetDb(ctx);
                const auto sol = solv.FindSolution(ctx, problem, db, {}, pdb_vals.at(val.first));
                assert(sol.Succeeded());
            }
            else
                assert(pdb_vals.find(val.first) ==
                       pdb_vals.end()); // << "Non-Tunable solver found in PDB";
            // If a solver used to be tunable and is no longer such, the pdb
            // entries should be removed to reclaim space in the db

            // It is possible that some perf db entries are not accessed in the map,
            // in which case we should remove them since they are taking up
            // space but are not useful.
        }
    }

    // for each key , val pair in the find db
    // for each solver entry in the val of find db
    // check that the solver is applicable
    // check the workspace value is correct
    // Make sure the perf db entry is valid
    // make sure the kdb entry exists and is loadable in hip
    // Bonus: No phantom entries in perf db, no phantom entries in kdb
    return 0;
}
