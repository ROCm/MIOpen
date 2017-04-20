#ifndef SOURCE_INLINER_HPP

#define SOURCE_INLINER_HPP
#include <stack>
#include <ostream>
#include "source_file_desc.hpp"

class InlineException : public std::exception {};

class InlineStackOverflowException : public InlineException
{
public:
    InlineStackOverflowException(const std::string& trace);
    char const* what() const override { return _trace.c_str(); }

private:
    std::string _trace;
};

class InlineFileNotFoundException : public InlineException
{
public:
    InlineFileNotFoundException(const std::string& file_name, const std::string& trace);
    char const* what() const override { return _trace.c_str(); }

private:
    std::string _trace;
};

class SourceInliner
{
public:
    int include_depth_limit = 256;

    void Process(std::istream& input, std::ostream& output, const std::string& root, const std::string& file_name);
    std::string GetIncludeStackTrace(int line);

private:
    int _include_depth = 0;
    SourceFileDesc* _included_stack_head = nullptr;

    void ProcessCore(std::istream& input, std::ostream& output, const std::string& root, const std::string& file_name, int line_number);
};

#endif // !SOURCE_INLINER_HPP
