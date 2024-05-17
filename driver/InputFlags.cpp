/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include "InputFlags.hpp"

#include "tensor_driver.hpp"

#include <miopen/tensor.hpp>
#include <miopen/stringutils.hpp>

#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>

int TensorParameters::SetTensordDescriptor(miopenTensorDescriptor_t result,
                                           miopenDataType_t data_type)
{
    if(layout.empty() && strides.empty())
        return SetTensorNd(result, lengths, data_type);

    if(strides.empty() && !layout.empty())
        CalculateStrides();

    return SetTensorNd(result, lengths, strides, data_type);
}

void TensorParameters::CalculateStrides()
{
    if(layout.empty())
        MIOPEN_THROW("Attempt to calculate strides without layout.");
    if(layout.size() != lengths.size())
        MIOPEN_THROW("Unmatched layout and lengths sizes.");

    const auto len_layout = miopen::tensor_layout_get_default(layout.size());
    if(len_layout.empty())
        MIOPEN_THROW("Invalid tensor lengths dimentions.");

    strides = {};
    miopen::tensor_layout_to_strides(lengths, len_layout, layout, strides);
}

InputFlags::InputFlags() { AddInputFlag("help", 'h', "", "Print Help Message", "string"); }

void InputFlags::AddInputFlag(const std::string& _long_name,
                              char _short_name,
                              const std::string& _value,
                              const std::string& _help_text,
                              const std::string& _type,
                              const bool _convert2uppercase)
{

    Input in;
    in.long_name         = _long_name;
    in.short_name        = _short_name;
    in.value             = _value;
    in.help_text         = _help_text;
    in.type              = _type;
    in.convert2uppercase = _convert2uppercase;

    if(MapInputs.count(_short_name) > 0)
        printf("Input flag: %s (%c) already exists !", _long_name.c_str(), _short_name);
    else
        MapInputs[_short_name] = in;
}

void InputFlags::AddTensorFlag(const std::string& name,
                               char short_name,
                               const std::string& default_value,
                               const std::string& default_desc)
{
    auto desc = std::ostringstream{};
    desc << static_cast<char>(std::toupper(name[0])) << name.substr(1) << " tensor descriptor."
         << std::endl;
    desc << "Format: NxC[xD]xHxW[,LayoutOrStrides]" << std::endl;
    desc << "Default: " << (default_desc.size() == 0 ? default_value : default_desc);

    AddInputFlag(name, short_name, default_value, desc.str(), "tensor descriptor");
}

void InputFlags::Print() const
{
    printf("MIOpen Driver Input Flags: \n\n");

    for(auto& content : MapInputs)
    {
        std::vector<std::string> help_text_lines;
        size_t pos = 0;
        for(size_t next_pos = content.second.help_text.find('\n', pos);
            next_pos != std::string::npos;)
        {
            help_text_lines.push_back(std::string(content.second.help_text.begin() + pos,
                                                  content.second.help_text.begin() + next_pos++));
            pos      = next_pos;
            next_pos = content.second.help_text.find('\n', pos);
        }
        help_text_lines.push_back(
            std::string(content.second.help_text.begin() + pos, content.second.help_text.end()));

        std::cout << std::setw(8) << "--" << content.second.long_name
                  << std::setw(20 - content.second.long_name.length()) << "-" << content.first
                  << std::setw(8) << " " << help_text_lines[0] << std::endl;

        for(auto help_next_line = std::next(help_text_lines.begin());
            help_next_line != help_text_lines.end();
            ++help_next_line)
        {
            std::cout << std::setw(37) << " " << *help_next_line << std::endl;
        }
    }
    exit(0); // NOLINT (concurrency-mt-unsafe)
}

char InputFlags::FindShortName(const std::string& long_name) const
{
    char short_name = '\0';
    for(auto& content : MapInputs)
    {
        if(content.second.long_name == long_name)
            short_name = content.first;
    }
    if(short_name == '\0')
    {
        std::cout << "Long Name: " << long_name << " Not Found !";
        exit(0); // NOLINT (concurrency-mt-unsafe)
    }
    return short_name;
}

void InputFlags::Parse(int argc, char* argv[])
{
    std::vector<std::string> args;
    for(int i = 2; i < argc; i++)
        args.push_back(argv[i]);

    //	if(args.size() == 0) // No Input Flag
    //		Print();

    for(int i = 0; i < args.size(); i++)
    {
        std::string temp = args[i];
        if(temp[0] != '-')
        {
            printf("Illegal input flag\n");
            Print();
        }
        else if(temp[0] == '-' && temp[1] == '-') // Long Name Input
        {
            std::string long_name = temp.substr(2);
            if(long_name == "help")
                Print();
            char short_name = FindShortName(long_name);
            StoreOptionalFlagValue(short_name, args[i + 1]);
            i++;
        }
        else if(temp[0] == '-' && temp[1] == '?') // Help Input
            Print();
        else // Short Name Input
        {
            char short_name = temp[1];
            if(MapInputs.find(short_name) == MapInputs.end())
            {
                std::cout << "Input Flag: " << short_name << " Not Found !";
                exit(0); // NOLINT (concurrency-mt-unsafe)
            }
            if(short_name == 'h')
                Print();

            if(i + 1 >= args.size()) // Check whether last arg has a value
                Print();
            else
            {
                MapInputs[short_name].value = args[i + 1];
                i++;
            }
        }
    }
}

// This function updates the input flag parameters values.Depending on the flag setting,
// input values are converted to uppercase & stored into map.This is used while
// parsing the driver arguments.
void InputFlags::StoreOptionalFlagValue(char short_name, const std::string& input_value)
{
    if(MapInputs[short_name].convert2uppercase == true)
    {
        std::string tvalue = input_value;
        std::transform(tvalue.begin(), tvalue.end(), tvalue.begin(), ::toupper);
        MapInputs[short_name].value = tvalue;
    }
    else
    {
        MapInputs[short_name].value = input_value;
    }
}

std::string InputFlags::GetValueStr(const std::string& long_name) const
{
    char short_name   = FindShortName(long_name);
    std::string value = MapInputs.at(short_name).value;

    return value;
}

int InputFlags::GetValueInt(const std::string& long_name) const
{
    char short_name = FindShortName(long_name);
    int value       = atoi(MapInputs.at(short_name).value.c_str());

    return value;
}

uint64_t InputFlags::GetValueUint64(const std::string& long_name) const
{
    char short_name = FindShortName(long_name);
    uint64_t value  = strtoull(MapInputs.at(short_name).value.c_str(), nullptr, 10);

    return value;
}

double InputFlags::GetValueDouble(const std::string& long_name) const
{
    char short_name = FindShortName(long_name);
    double value    = atof(MapInputs.at(short_name).value.c_str());

    return value;
}

TensorParameters InputFlags::GetValueTensor(const std::string& long_name) const
{
    const auto& input     = MapInputs.at(FindShortName(long_name));
    const auto components = miopen::SplitDelim(input.value.c_str(), ',');

    if(components.size() < 1)
        return {};

    auto parse = [](auto line) {
        auto ret        = std::vector<int>{};
        const auto strs = miopen::SplitDelim(line, 'x');
        for(auto&& str : strs)
        {
            auto elem = int{};
            auto ss   = std::istringstream{str};
            ss >> elem;

            if(ss.bad() || ss.fail())
                MIOPEN_THROW("Invalid tensor component " + str + " in " + line + ".");

            ret.push_back(elem);
        }
        return ret;
    };

    auto lens = parse(components[0]);

    if(components.size() == 1)
        return {lens};

    auto layout  = std::string{};
    auto strides = std::vector<int>{};

    if(std::isdigit(components[1][0]))
        strides = parse(components[1]);
    else
        layout = components[1];

    if(components.size() == 2)
        return {lens, strides, layout};

    MIOPEN_THROW("Too many tensor descriptor parameters.");
}

TensorParametersUint64 InputFlags::GetValueTensorUint64(const std::string& long_name) const
{
    const auto& input     = MapInputs.at(FindShortName(long_name));
    const auto components = miopen::SplitDelim(input.value.c_str(), ',');

    if(components.size() < 1)
        return {};

    auto parse = [](auto line) {
        auto ret        = std::vector<uint64_t>{};
        const auto strs = miopen::SplitDelim(line, 'x');
        for(auto&& str : strs)
        {
            auto elem = uint64_t{};
            auto ss   = std::istringstream{str};
            ss >> elem;

            if(ss.bad() || ss.fail())
                MIOPEN_THROW("Invalid tensor component " + str + " in " + line + ".");

            ret.push_back(elem);
        }
        return ret;
    };

    auto lens = parse(components[0]);

    if(components.size() == 1)
        return {lens};

    auto layout  = std::string{};
    auto strides = std::vector<uint64_t>{};

    if(std::isdigit(components[1][0]))
        strides = parse(components[1]);
    else
        layout = components[1];

    if(components.size() == 2)
        return {lens, strides, layout};

    MIOPEN_THROW("Too many tensor descriptor parameters.");
}

std::vector<int32_t> InputFlags::GetValueVectorInt(const std::string& long_name) const
{
    const auto& input = MapInputs.at(FindShortName(long_name));

    auto ret        = std::vector<int32_t>{};
    const auto strs = miopen::SplitDelim(input.value.c_str(), ',');

    for(auto&& str : strs)
    {
        auto elem = int32_t{};
        auto ss   = std::istringstream{str};
        ss >> elem;

        if(ss.bad() || ss.fail())
            MIOPEN_THROW("Invalid tensor component " + str + " in " + input.value.c_str() + ".");

        ret.push_back(elem);
    }

    return ret;
}

std::vector<uint64_t> InputFlags::GetValueVectorUint64(const std::string& long_name) const
{
    const auto& input = MapInputs.at(FindShortName(long_name));

    auto ret        = std::vector<uint64_t>{};
    const auto strs = miopen::SplitDelim(input.value.c_str(), ',');

    for(auto&& str : strs)
    {
        auto elem = uint64_t{};
        auto ss   = std::istringstream{str};
        ss >> elem;

        if(ss.bad() || ss.fail())
            MIOPEN_THROW("Invalid tensor component " + str + " in " + input.value.c_str() + ".");

        ret.push_back(elem);
    }

    return ret;
}

std::vector<std::vector<int32_t>>
InputFlags::GetValue2dVectorInt(const std::string& long_name) const
{
    const auto& input     = MapInputs.at(FindShortName(long_name));
    const auto components = miopen::SplitDelim(input.value.c_str(), ',');
    auto output           = std::vector<std::vector<int32_t>>{};

    if(components.size() < 1)
        return {};

    auto parse = [](auto line) {
        auto ret        = std::vector<int32_t>{};
        const auto strs = miopen::SplitDelim(line, 'x');
        for(auto&& str : strs)
        {
            auto elem = int32_t{};
            auto ss   = std::istringstream{str};
            ss >> elem;

            if(ss.bad() || ss.fail())
                MIOPEN_THROW("Invalid tensor component " + str + " in " + line + ".");

            ret.push_back(elem);
        }
        return ret;
    };

    for(auto&& component : components)
    {
        output.push_back(parse(component));
    }

    return output;
}

std::vector<std::vector<uint64_t>>
InputFlags::GetValue2dVectorUint64(const std::string& long_name) const
{
    const auto& input     = MapInputs.at(FindShortName(long_name));
    const auto components = miopen::SplitDelim(input.value.c_str(), ',');
    auto output           = std::vector<std::vector<uint64_t>>{};

    if(components.size() < 1)
        return {};

    auto parse = [](auto line) {
        auto ret        = std::vector<uint64_t>{};
        const auto strs = miopen::SplitDelim(line, 'x');
        for(auto&& str : strs)
        {
            auto elem = uint64_t{};
            auto ss   = std::istringstream{str};
            ss >> elem;

            if(ss.bad() || ss.fail())
                MIOPEN_THROW("Invalid tensor component " + str + " in " + line + ".");

            ret.push_back(elem);
        }
        return ret;
    };

    for(auto&& component : components)
    {
        output.push_back(parse(component));
    }

    return output;
}

void InputFlags::SetValue(const std::string& long_name, const std::string& new_value)
{
    char short_name                = FindShortName(long_name);
    MapInputs.at(short_name).value = new_value;
}
