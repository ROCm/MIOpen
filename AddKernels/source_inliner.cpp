#include <exception>
#include <sstream>
#include <fstream>

#ifdef WIN32
#include <windows.h>
#else
#include <stdlib.h>
#include <linux/limits.h>
#endif // !WIN32

#include "source_inliner.hpp"

namespace PathHelpers
{
    static int GetMaxPath()
    {
#ifdef WIN32
        return MAX_PATH;
#else
        return PATH_MAX;
#endif
    }

    static std::string GetAbsolutePath(const std::string& path)
    {
        std::string result(GetMaxPath(), ' ');
#ifdef WIN32
        const auto retval = GetFullPathName(path.c_str(),
            result.size(),
            &result[0],
            nullptr);

        if (retval == 0)
            return "";
#else
        const auto retval = realpath(path.c_str(), &result[0]);

        if (retval == nullptr)
            return "";
#endif
        return result;
    }
}

InlineStackOverflowException::InlineStackOverflowException(const std::string& trace)
{
    std::ostringstream ss;

    ss << "Include stack depth limit has been reached." << std::endl;
    ss << trace;

    _trace = ss.str();
}

InlineFileNotFoundException::InlineFileNotFoundException(const std::string& file_name, const std::string& trace)
{
    std::ostringstream ss;

    ss << "Include file was not found or can't be opened: <" << file_name << ">." << std::endl;
    ss << trace;

    _trace = ss.str();
}

InlineWrongIncludeFormatException::InlineWrongIncludeFormatException(const std::string& file_name, const std::string& trace)
{
    std::ostringstream ss;

    ss << "Wron format of include directive in file <" << file_name << ">. Expected:" << std::endl;
    ss << ".include \"path to include\"" << std::endl;
    ss << trace;

    _trace = ss.str();
}

void SourceInliner::Process(std::istream& input, std::ostream& output, const std::string& root, const std::string& file_name)
{
    ProcessCore(input, output, root, file_name, 0);
}

void SourceInliner::ProcessCore(std::istream& input, std::ostream& output, const std::string& root, const std::string& file_name, int line_number)
{
    if (_include_depth >= include_depth_limit)
        throw InlineStackOverflowException(GetIncludeStackTrace(0));

    _include_depth++;
    _included_stack_head = new SourceFileDesc(file_name, _included_stack_head, line_number);
    auto current_line = 0;

    while (!input.eof())
    {
        std::string line, word;
        std::getline(input, line);
        std::istringstream line_parser(line);
        line_parser >> word;
        current_line++;

        if (word == ".include")
        {
            const auto first_quote_pos = line.find('"', (int)line_parser.tellg() + 1);
            if (first_quote_pos == std::string::npos)
                throw InlineWrongIncludeFormatException(file_name, GetIncludeStackTrace(current_line));

            const auto second_quote_pos = line.find('"', first_quote_pos + 1);
            if (second_quote_pos == std::string::npos)
                throw InlineWrongIncludeFormatException(file_name, GetIncludeStackTrace(current_line));

            const std::string include_file_path = line.substr(first_quote_pos + 1, second_quote_pos - first_quote_pos - 1);
            const std::string abs_include_file_path(PathHelpers::GetAbsolutePath(root + "/" + include_file_path));
            std::ifstream include_file(abs_include_file_path, std::ios::in);

            if (!include_file.good())
                throw InlineFileNotFoundException(include_file_path, GetIncludeStackTrace(current_line));

            ProcessCore(include_file, output, root, include_file_path, current_line);
        }
        else
        {
            if (output.tellp() > 0)
                output << std::endl;

            output << line;
        }
    }

    auto prev_file = _included_stack_head->included_from;
    delete _included_stack_head;
    _included_stack_head = prev_file;
    _include_depth--;
}

std::string SourceInliner::GetIncludeStackTrace(int line)
{
    std::ostringstream ss;

    if (_included_stack_head == nullptr)
        return "";

    auto item = _included_stack_head;
    ss << "    " << item->path << ":" << line;

    while (item->included_from != nullptr)
    {
        ss << std::endl << "    from " << item->included_from->path << ":" << item->included_line;
        item = item->included_from;
    }

    return ss.str();
}
