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
#include "include_inliner.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

void Bin2Hex(std::istream& source,
             std::ostream& target,
             const std::string& variable,
             bool nullTerminate,
             size_t bufferSize,
             size_t lineSize)
{
    source.seekg(0, std::ios::end);
    std::unique_ptr<unsigned char[]> buffer(new unsigned char[bufferSize]);
    std::streamoff sourceSize = source.tellg();
    std::streamoff blockStart = 0;

    if(variable.length() != 0)
    {
        target << "extern const size_t " << variable << "_SIZE;" << std::endl;
        target << "extern const unsigned char " << variable << "[];" << std::endl;
        target << "const size_t " << variable << "_SIZE = " << std::setbase(10) << sourceSize << ";"
               << std::endl;
        target << "const unsigned char " << variable << "[] = {" << std::endl;
    }

    target << std::setbase(16) << std::setfill('0');
    source.seekg(0, std::ios::beg);

    while(blockStart < sourceSize)
    {
        source.read(reinterpret_cast<char*>(buffer.get()), bufferSize);

        std::streamoff pos       = source.tellg();
        std::streamoff blockSize = (pos < 0 ? sourceSize : pos) - blockStart;
        std::streamoff i         = 0;

        while(i < blockSize)
        {
            size_t j   = i;
            size_t end = std::min<size_t>(i + lineSize, blockSize);

            for(; j < end; j++)
                target << "0x" << std::setw(2) << static_cast<unsigned>(buffer[j]) << ",";

            target << std::endl;
            i = end;
        }

        blockStart += blockSize;
    }

    if(nullTerminate)
        target << "0x00," << std::endl;

    if(variable.length() != 0)
    {
        target << "};" << std::endl;
    }
}

void PrintHelp()
{
    std::cout << "Usage: bin2hex {<option>}" << std::endl;
    std::cout << "Option format: -<option name>[ <option value>]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout
        << "[REQUIRED] -s[ource] {<path to file>}: files to be processed. Must be last argument."
        << std::endl;
    std::cout << "           -t[arget] <path>: target file. Default: std out." << std::endl;
    std::cout << "           -l[ine-size] <number>: bytes in one line. Default: 16." << std::endl;
    std::cout << "           -b[uffer] <number>: read buffer size. Default: 512." << std::endl;
    std::cout << "           -g[uard] <string>: guard name. Default: no guard" << std::endl;
    std::cout << "           -n[o-recurse] : dont expand include files recursively. Default: off"
              << std::endl;
}

[[gnu::noreturn]] void WrongUsage(const std::string& error)
{
    std::cout << "Wrong usage: " << error << std::endl;
    std::cout << std::endl;
    PrintHelp();
    std::exit(1);
}

[[gnu::noreturn]] void UnknownArgument(const std::string& arg)
{
    std::ostringstream ss;
    ss << "unknown argument - " << arg;
    WrongUsage(ss.str());
}

void Process(const std::string& sourcePath,
             std::ostream& target,
             size_t bufferSize,
             size_t lineSize,
             bool recurse,
             bool as_extern)
{
    std::string fileName(sourcePath);
    std::string extension, root;
    std::stringstream inlinerTemp;
    auto extPos   = fileName.rfind('.');
    auto slashPos = fileName.rfind('/');

    if(extPos != std::string::npos)
    {
        extension = fileName.substr(extPos + 1);
        fileName  = fileName.substr(0, extPos);
    }

    if(slashPos != std::string::npos)
    {
        root     = fileName.substr(0, slashPos + 1);
        fileName = fileName.substr(slashPos + 1);
    }

    std::string variable(fileName);
    std::ifstream sourceFile(sourcePath, std::ios::in | std::ios::binary);
    std::istream* source = &sourceFile;

    if(!sourceFile.good())
    {
        std::cerr << "File not found: " << sourcePath << std::endl;
        std::exit(1);
    }

    const auto is_asm    = extension == "s";
    const auto is_cl     = extension == "cl";
    const auto is_hip    = extension == "cpp";
    const auto is_header = extension == "hpp";

    if(is_asm || is_cl || is_hip || is_header)
    {
        IncludeInliner inliner;

        try
        {
            if(is_asm)
                inliner.Process(
                    sourceFile, inlinerTemp, root, sourcePath, ".include", false, recurse);
            else if(is_cl || is_header)
                inliner.Process(
                    sourceFile, inlinerTemp, root, sourcePath, "#include", true, recurse);
            else if(is_hip)
                inliner.Process(
                    sourceFile, inlinerTemp, root, sourcePath, "<#not_include>", true, false);
        }
        catch(const InlineException& ex)
        {
            std::cerr << ex.What() << std::endl;
            std::cerr << ex.GetTrace() << std::endl;
            std::exit(1);
        }

        source = &inlinerTemp;
    }

    std::transform(variable.begin(), variable.end(), variable.begin(), ::toupper);

    if(as_extern && variable.length() != 0)
    {
        variable = "APP_KERNEL_" + variable;
    }

    Bin2Hex(*source, target, variable, true, bufferSize, lineSize);
}

int main(int argsn, char** args)
{
    if(argsn == 1)
    {
        PrintHelp();
        return 2;
    }

    std::string guard;
    size_t bufferSize = 512;
    size_t lineSize   = 16;

    std::ofstream targetFile;
    std::ostream* target = &std::cout;
    bool recurse         = true;
    bool as_extern       = false;

    int i = 0;
    while(++i < argsn && **args != '-')
    {
        std::string arg(args[i] + 1);
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);

        if(arg == "s" || arg == "source")
        {
            if(guard.length() > 0)
            {
                *target << "#ifndef " << guard << std::endl;
                *target << "#define " << guard << std::endl;
            }

            *target << "#include <cstddef>" << std::endl;

            while(++i < argsn)
            {
                Process(args[i], *target, bufferSize, lineSize, recurse, as_extern);
            }

            if(guard.length() > 0)
            {
                *target << "#endif" << std::endl;
            }

            return 0;
        }
        else if(arg == "t" || arg == "target")
        {
            targetFile.open(args[++i], std::ios::out);
            target = &targetFile;
        }
        else if(arg == "l" || arg == "line-size")
            lineSize = std::stol(args[++i]);
        else if(arg == "b" || arg == "buffer")
            bufferSize = std::stol(args[++i]);
        else if(arg == "g" || arg == "guard")
            guard = args[++i];
        else if(arg == "n" || arg == "no-recurse")
            recurse = false;
        else if(arg == "e" || arg == "extern")
            as_extern = true;
        else
            UnknownArgument(arg);
    }

    WrongUsage("source key is required");
}
