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
#ifndef SOURCE_INLINER_HPP

#define SOURCE_INLINER_HPP
#include "source_file_desc.hpp"
#include <ostream>
#include <memory>
#include <stack>

class InlineException : public std::exception
{
    public:
    char const* what() const noexcept override { return _trace.c_str(); }

    protected:
    std::string _trace;
};

class InlineStackOverflowException : public InlineException
{
    public:
    InlineStackOverflowException(const std::string& trace);
};

class IncludeInliner
{
    public:
    int include_depth_limit = 256;

    void Process(std::istream& input,
                 std::ostream& output,
                 const std::string& root,
                 const std::string& file_name);
    std::string GetIncludeStackTrace(int line);

    private:
    int _include_depth                                   = 0;
    std::shared_ptr<SourceFileDesc> _included_stack_head = nullptr;

    void ProcessCore(std::istream& input,
                     std::ostream& output,
                     const std::string& root,
                     const std::string& file_name,
                     int line_number);
};

#endif // !SOURCE_INLINER_HPP
