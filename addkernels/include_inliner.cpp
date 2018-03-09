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
#include <algorithm>
#include <exception>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif
#ifdef __linux__
#include <linux/limits.h>
#include <cstdlib>
#endif // !WIN32

#include "include_inliner.hpp"

namespace PathHelpers {
static int GetMaxPath()
{
#ifdef _WIN32
    return MAX_PATH;
#else
    return PATH_MAX;
#endif
}

static std::string GetAbsolutePath(const std::string& path)
{
    std::string result(GetMaxPath(), ' ');
#ifdef _WIN32
    const auto retval = GetFullPathName(path.c_str(), result.size(), &result[0], nullptr);

    if(retval == 0)
        return "";
#else
    const auto retval = realpath(path.c_str(), &result[0]);

    if(retval == nullptr)
        return "";
#endif
    return result;
}
} // namespace PathHelpers

InlineStackOverflowException::InlineStackOverflowException(const std::string& trace)
{
    std::ostringstream ss;

    ss << "Include stack depth limit has been reached." << std::endl;
    ss << trace;

    _trace = ss.str();
}

void IncludeInliner::Process(std::istream& input,
                             std::ostream& output,
                             const std::string& root,
                             const std::string& file_name)
{
    ProcessCore(input, output, root, file_name, 0);
}

void IncludeInliner::ProcessCore(std::istream& input,
                                 std::ostream& output,
                                 const std::string& root,
                                 const std::string& file_name,
                                 int line_number)
{
    if(_include_depth >= include_depth_limit)
        throw InlineStackOverflowException(GetIncludeStackTrace(0));

    _include_depth++;
    _included_stack_head =
        std::make_shared<SourceFileDesc>(file_name, _included_stack_head, line_number);
    auto current_line = 0;

    while(!input.eof())
    {
        std::string line, word;
        std::getline(input, line);
        std::istringstream line_parser(line);
        line_parser >> word;
        current_line++;
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        if(!word.empty() && word == ".include")
        {
            const auto first_quote_pos = line.find('"', static_cast<int>(line_parser.tellg()) + 1);
            if(first_quote_pos == std::string::npos)
                continue;

            const auto second_quote_pos = line.find('"', first_quote_pos + 1);
            if(second_quote_pos == std::string::npos)
                continue;

            const std::string include_file_path =
                line.substr(first_quote_pos + 1, second_quote_pos - first_quote_pos - 1);
            const std::string abs_include_file_path(
                PathHelpers::GetAbsolutePath(root + "/" + include_file_path)); // NOLINT
            std::ifstream include_file(abs_include_file_path, std::ios::in);

            if(!include_file.good())
                continue;

            ProcessCore(include_file, output, root, include_file_path, current_line);
        }
        else
        {
            if(output.tellp() > 0)
                output << std::endl;

            output << line;
        }
    }

    auto prev_file       = _included_stack_head->included_from;
    _included_stack_head = prev_file;
    _include_depth--;
}

std::string IncludeInliner::GetIncludeStackTrace(int line)
{
    std::ostringstream ss;

    if(_included_stack_head == nullptr)
        return "";

    auto item = _included_stack_head;
    ss << "    " << item->path << ":" << line;

    while(item->included_from != nullptr)
    {
        ss << std::endl << "    from " << item->included_from->path << ":" << item->included_line;
        item = item->included_from;
    }

    return ss.str();
}
