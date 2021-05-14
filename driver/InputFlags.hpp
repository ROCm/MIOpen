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
#ifndef MIOPEN_INPUT_FLAGS_HPP_
#define MIOPEN_INPUT_FLAGS_HPP_

#include <map>
#include <string>

struct Input
{
    std::string long_name;
    char short_name;
    std::string value;
    std::string help_text;
    std::string type;
};

class InputFlags
{
    std::map<char, Input> MapInputs;

    public:
    InputFlags();
    void AddInputFlag(const std::string& _long_name,
                      char _short_name,
                      const std::string& _value,
                      const std::string& _help_text,
                      const std::string& type);
    void Parse(int argc, char* argv[]);
    char FindShortName(const std::string& _long_name) const;
    void Print() const;

    std::string GetValueStr(const std::string& _long_name) const;
    int GetValueInt(const std::string& _long_name) const;
    uint64_t GetValueUint64(const std::string& _long_name) const;
    double GetValueDouble(const std::string& _long_name) const;
    void SetValue(const std::string& long_name, const std::string& new_value);

    virtual ~InputFlags() {}
};

#endif //_MIOPEN_INPUT_FLAGS_HPP_
