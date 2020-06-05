/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include "sqlite3.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <iterator>
#include <unordered_map>
#include <numeric>

void Usage()
{
    std::cout
        << "addDbs {--sqlite | --plaintext } --arch <arch-cu> --output <filename> --input <db file>"
        << std::endl;
}

using result_type = std::vector<std::unordered_map<std::string, std::string>>;
static int find_callback(void* _res, int argc, char** argv, char** azColName)
{
    result_type* res = static_cast<result_type*>(_res);
    std::unordered_map<std::string, std::string> record;
    for(auto i               = 0; i < argc; i++)
        record[azColName[i]] = (argv[i] != nullptr) ? argv[i] : "NULL";
    if(res != nullptr)
        res->push_back(record);
    return 0;
}

template <typename Out>
void split(const std::string& s, char delim, Out result)
{
    std::istringstream iss(s);
    std::string item;
    while(std::getline(iss, item, delim))
    {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

int main(int argc, char* argv[])
{
    bool is_sqlite = false;
    std::vector<boost::filesystem::path> input_files;
    boost::filesystem::path output_file;
    std::string arch_cu;

    auto idx = 1;
    while(idx < argc)
    {
        std::string cur_arg = argv[idx];
        if(cur_arg == "--sqlite")
            is_sqlite = true;
        else if(cur_arg == "--plaintext")
            is_sqlite = false;
        else if(cur_arg == "--output")
            output_file = boost::filesystem::path(argv[++idx]);
        else if(cur_arg == "--arch")
            arch_cu = std::string(argv[++idx]);
        else
            input_files.emplace_back(boost::filesystem::path(argv[idx]));
        ++idx;
    }
    auto lst_arches = split(arch_cu, ':');
    std::vector<std::pair<std::string, std::string>> arches;
    for(auto& ac : lst_arches)
    {
        auto toks = split(ac, '_');
        if(toks.size() != 2)
        {
            std::cerr << "Invalid arch/cu pair " << ac << std::endl;
            std::cerr << "Expected format:  [arch]_[num_cu] " << std::endl;
            exit(-1);
        }
        auto arch   = toks[0];
        auto num_cu = toks[1];
        arches.emplace_back(arch, num_cu);
    }
    if(is_sqlite)
    {
        if(input_files.size() > 1)
        {
            std::cerr << "SQLite DBs require only one input file" << std::endl;
            exit(-1);
        }
        auto input_file  = input_files[0];
        sqlite3* ptr_sql = nullptr;
        auto rc = sqlite3_open_v2(input_file.c_str(), &ptr_sql, SQLITE_OPEN_READONLY, nullptr);
        if(rc != SQLITE_OK)
        {
            std::cerr << "Unable to open database file: " << input_file << std::endl;
            exit(-1);
        }
        std::ofstream ss(output_file.string(), std::ios::out);
        if(!ss.good())
        {
            std::cerr << "Unable to open output file: " << output_file.string() << std::endl;
            exit(-1);
        }
        for(auto arch_idx = 0; arch_idx < arches.size(); arch_idx++)
        {
            auto kinder = arches[arch_idx];
            auto arch   = kinder.first;
            auto num_cu = kinder.second;
            std::stringstream query_ss;
            query_ss << "select key, group_concat(solver || ':' || params, ';') as res "
                        "from config inner join perf_db on config.id = perf_db.config "
                        "where arch = '"
                     << arch << "' and num_cu = " << num_cu << " group by key";
            result_type configs;
            rc = sqlite3_exec(ptr_sql,
                              query_ss.str().c_str(),
                              find_callback,
                              static_cast<void*>(&configs),
                              nullptr);
            if(rc != SQLITE_OK)
            {
                std::cerr << "Unable to execute query: " << query_ss.str() << std::endl;
                exit(-1);
            }
            if(arch_idx != 0)
                ss << " else ";

            ss << "if(arch_cu == \"" << arch << "_" << num_cu << "\") \n {\n static const "
                                                                 "std::unordered_map<std::string, "
                                                                 "std::string> data \n\t {\n";
            // handle the joining comma
            ss << "\t\t { \"" << configs[0]["key"] << "\", \"" << configs[0]["res"] << "\"}"
               << std::endl;
            for(auto cfg_idx = 1; cfg_idx < configs.size(); cfg_idx++)
            {
                auto config = configs[cfg_idx];
                if(!config["res"].empty())
                    ss << "\t\t,{ \"" << config["key"] << "\", \"" << config["res"] << "\"}"
                       << std::endl;
            }
            ss << "};\n return data; \n}" << std::endl;
        }
        // default case for invalid/unknown arch-cu pair
        ss << "else {\n static const std::unordered_map<std::string, std::string> data \n\t {\n}; "
              "\n return data; \n}"
           << std::endl;
    }
    else // plaintext
    {

        auto line = std::string{};
        std::ofstream ss(output_file.string(), std::ios::out);
        if(!ss.good())
        {
            std::cerr << "Unable to open output file: " << output_file.string() << std::endl;
            exit(-1);
        }
        for(auto arch_idx = 0; arch_idx < arches.size(); arch_idx++)
        {
            auto kinder = arches[arch_idx];
            auto arch   = kinder.first;
            auto num_cu = kinder.second;

            boost::filesystem::path input_file;
            for(auto& tmp : input_files)
            {
                std::string str_file = boost::filesystem::basename(tmp);
                std::stringstream tmp_ss;
                tmp_ss << arch << "_" << num_cu;
                if(str_file.find(tmp_ss.str()) != std::string::npos)
                {
                    input_file = tmp;
                    break;
                }
            }
            auto file = std::ifstream{input_file.string()};
            if(!file)
            {
                std::cerr << "Unable to open input file: " << input_file.string() << std::endl;
                exit(-1);
            }
            if(arch_idx != 0)
                ss << " else ";

            ss << "if(arch_cu == \"" << arch << "_" << num_cu << "\") \n {\n static const "
                                                                 "std::unordered_map<std::string, "
                                                                 "std::string> data \n\t {\n";
            auto line_cnt = 0;
            while(std::getline(file, line))
            {
                if(line.empty())
                    continue;

                const auto key_size = line.find('=');
                const bool is_key   = (key_size != std::string::npos && key_size != 0);
                if(!is_key)
                {
                    std::cerr << "Ill formed key: " << line << std::endl;
                    continue;
                }
                const auto key      = line.substr(0, key_size);
                const auto contents = line.substr(key_size + 1);
                if(line_cnt != 0)
                    ss << ",";
                ss << "\t\t{ \"" << key << "\", \"" << contents << "\"}" << std::endl;
                line_cnt++;
            }
            ss << "};\n return data; \n}" << std::endl;
        }
        ss << "else {\n static const std::unordered_map<std::string, std::string> data \n\t {\n}; "
              "\n return data; \n}"
           << std::endl;
    }
    return 0;
}
