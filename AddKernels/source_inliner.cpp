#include <exception>
#include <sstream>
#include <fstream>
#include "source_inliner.hpp"

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

void SourceInliner::Process(std::istream& input, std::ostream& output, const std::string& root, const std::string& file_name)
{
    ProcessCore(input, output, root, file_name, 0);
}

void SourceInliner::ProcessCore(std::istream& input, std::ostream& output, const std::string& root, const std::string& file_name, int line_number)
{
    if (_include_depth >= include_depth_limit)
        throw InlineStackOverflowException(GetIncludeStackTrace());

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
            std::string include_file_path = line.substr((int)line_parser.tellg() + 1);
            include_file_path.pop_back();
            std::string abs_include_file_path = root + "/" + include_file_path; // TODO get abs path
            std::ifstream include_file(abs_include_file_path, std::ios::in);

            if (include_file.bad())
                throw InlineFileNotFoundException(include_file_path, GetIncludeStackTrace());

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

std::string SourceInliner::GetIncludeStackTrace()
{
    std::ostringstream ss;

    if (_included_stack_head == nullptr)
        return "";

    auto item = _included_stack_head;
    ss << "    " << item->path;

    while (item->included_from != nullptr)
    {
        ss << std::endl << "    from " << item->included_from->path << ":" << item->included_line;
        item = item->included_from;
    }

    return ss.str();
}
