// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <sqlite3.h>

#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <unordered_map>

std::unique_ptr<sqlite3, int (*)(sqlite3*)> OpenDb(const char* filename, int flags)
{
    sqlite3* db;
    if(sqlite3_open_v2(filename, &db, flags, nullptr) != SQLITE_OK)
        abort();
    if(db == nullptr)
        abort();
    return {db, &sqlite3_close_v2};
}

std::unique_ptr<sqlite3_stmt, int (*)(sqlite3_stmt*)> PrepareStatement(sqlite3* db,
                                                                       const std::string& sql)
{
    sqlite3_stmt* stmt;
    const char* tail;
    if(sqlite3_prepare_v2(db, sql.c_str(), sql.length(), &stmt, &tail) != SQLITE_OK ||
       stmt == nullptr)
    {
        std::cerr << "Error while preparing SQL statement: " << sqlite3_errmsg(db) << std::endl;
        std::cerr << "Statement: {" << sql << "}" << std::endl;
        abort();
    }
    if(tail != &sql[0] + sql.length())
    {
        std::cerr << "Statement leftover: {" << tail << "}" << std::endl;
        abort();
    }
    return {stmt, &sqlite3_finalize};
}

struct ProblemConfig
{
    int64_t in_d, in_h, in_w;
    int64_t fil_d, fil_h, fil_w;
    int64_t pad_d, pad_h, pad_w;
    int64_t conv_stride_d, conv_stride_h, conv_stride_w;
    int64_t dilation_d, dilation_h, dilation_w;
    int64_t spatial_dim, out_channels, in_channels, batchsize, group_count, bias;
    std::string layout, data_type, direction;

    template <class Self>
    static void Visit(Self&& self, std::function<void(int64_t&, std::string)> f)
    {
        // The column names match the driver command line argument names
        f(self.spatial_dim, "spatial_dim");
        f(self.in_channels, "in_channels");
        f(self.in_h, "in_h");
        f(self.in_w, "in_w");
        f(self.in_d, "in_d");
        f(self.fil_h, "fil_h");
        f(self.fil_w, "fil_w");
        f(self.fil_d, "fil_d");
        f(self.out_channels, "out_channels");
        f(self.batchsize, "batchsize");
        f(self.pad_h, "pad_h");
        f(self.pad_w, "pad_w");
        f(self.pad_d, "pad_d");
        f(self.conv_stride_h, "conv_stride_h");
        f(self.conv_stride_w, "conv_stride_w");
        f(self.conv_stride_d, "conv_stride_d");
        f(self.dilation_h, "dilation_h");
        f(self.dilation_w, "dilation_w");
        f(self.dilation_d, "dilation_d");
        f(self.bias, "bias");
        f(self.group_count, "group_count");
    }

    template <class Self>
    static void Visit(Self&& self, std::function<void(std::string&, std::string)> f)
    {
        f(self.layout, "layout");
        f(self.data_type, "data_type");
        f(self.direction, "direction");
    }

    template <class Self, class Visitor>
    static void VisitAll(Self&& self, const Visitor& f)
    {
        Visit(std::forward<Self>(self), [&](int64_t& value, std::string name) { f(value, name); });
        Visit(std::forward<Self>(self),
              [&](std::string& value, std::string name) { f(value, name); });
    }

    [[nodiscard]] static const std::string& GetFieldNames()
    {
        static const std::string value = []() {
            std::ostringstream ss;
            ProblemConfig::VisitAll(ProblemConfig{}, [&](auto&&, auto name) {
                if(ss.tellp() != 0)
                    ss << ", ";
                ss << name;
            });
            return ss.str();
        }();
        return value;
    }

    [[nodiscard]] std::string Serialize()
    {
        std::ostringstream ss;
        ProblemConfig::VisitAll(*this, [&](auto&& value, auto&&) {
            if(ss.tellp() != 0)
                ss << "x";
            ss << value;
        });
        return ss.str();
    }
};

int main(int argn, char** args)
{
    if(argn < 2 || argn > 3)
    {
        std::cerr << "Usage:" << std::endl;
        std::cerr << args[0] << " input_path [output_path]" << std::endl;
        std::cerr << "input_path - path to the input file, expected to be sqlite3 db." << std::endl;
        std::cerr << "output_path - optional path to the output file. Existing file would be "
                     "replaced. Defaults to the input_path with .txt appended to the end"
                  << std::endl;
    }

    const std::string in_filename  = args[1];
    const std::string out_filename = argn > 2 ? args[2] : (in_filename + ".txt");
    constexpr const int db_flags   = SQLITE_OPEN_READONLY;

    const auto select_query = "SELECT solver, params, " + ProblemConfig::GetFieldNames() +
                              " FROM perf_db "
                              "INNER JOIN config ON perf_db.config = config.id";

    const auto db   = OpenDb(in_filename.c_str(), db_flags);
    const auto stmt = PrepareStatement(db.get(), select_query);
    auto db_content = std::unordered_map<std::string, std::string>{};

    for(int step_result = sqlite3_step(stmt.get()); step_result != SQLITE_DONE;
        step_result     = sqlite3_step(stmt.get()))
    {
        if(step_result == SQLITE_BUSY)
        {
            sqlite3_sleep(10);
            continue;
        }

        if(step_result == SQLITE_ERROR)
        {
            std::cerr << sqlite3_errmsg(db.get()) << std::endl;
            abort();
        }

        if(step_result == SQLITE_MISUSE)
            abort();

        int col             = 0;
        std::string solver  = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), col++));
        std::string perfcgf = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), col++));
        ProblemConfig problem;

        ProblemConfig::VisitAll(problem, [&](auto& value, auto) {
            if constexpr(std::is_convertible_v<decltype(value), int>)
                value = sqlite3_column_int(stmt.get(), col++);
            else if constexpr(std::is_convertible_v<decltype(value), std::string>)
                value = reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), col++));
            else
                static_assert(false, "unsupported type");
        });

        if(sqlite3_column_count(stmt.get()) != col)
            abort();

        auto& record = db_content[problem.Serialize()];
        if(!record.empty())
            record.append(";");
        record.append(solver).append(":").append(perfcgf);
    }

    auto out = std::ofstream{out_filename};
    for(const auto& line : db_content)
        out << line.first << "=" << line.second << std::endl;

    return 0;
}
