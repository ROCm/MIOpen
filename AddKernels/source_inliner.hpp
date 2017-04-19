#ifndef SOURCE_INLINER_HPP

#define SOURCE_INLINER_HPP
#include <stack>
#include <ostream>
#include "source_file_desc.hpp"

class InlineStackOverflowException : std::exception
{
public:
    InlineStackOverflowException(const std::string& trace);
    char const* what() const override;

private:
    std::string _trace;
};

class SourceInliner
{
public:
    int include_depth_limit = 256;

    void Process(std::istream& input, std::ostream& output, const std::string& root, const std::string& file_name);
    std::string GetIncludeStackTrace();

private:
    int _include_depth = 0;
    SourceFileDesc* _included_stack_head = nullptr;

    void ProcessCore(std::istream& input, std::ostream& output, const std::string& root, const std::string& file_name, int line_number);
};

#endif // !SOURCE_INLINER_HPP
