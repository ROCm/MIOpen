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
#include "include_inliner.hpp"
#include "miopen/filesystem.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>

namespace fs = miopen::fs;

void Bin2Hex(std::istream& source,
             std::ostream& target,
             const std::string& variable,
             bool nullTerminate,
             size_t bufferSize,
             size_t lineSize)
{
    source.seekg(0, std::ios::end);
    const std::unique_ptr<unsigned char[]> buffer(new unsigned char[bufferSize]);
    const std::streamoff sourceSize = source.tellg();
    std::streamoff blockStart       = 0;

    if(variable.length() != 0)
    {
        target << "extern const size_t " << variable << "_SIZE;" << std::endl;
        target << "extern const char " << variable << "[];" << std::endl;
        target << "const size_t " << variable << "_SIZE = " << std::setbase(10) << sourceSize << ";"
               << std::endl;
        target << "const char " << variable << "[] = {" << std::endl;
    }

    target << std::setbase(16) << std::setfill('0');
    source.seekg(0, std::ios::beg);

    while(blockStart < sourceSize)
    {
        source.read(reinterpret_cast<char*>(buffer.get()), bufferSize);

        const std::streamoff pos       = source.tellg();
        const std::streamoff blockSize = (pos < 0 ? sourceSize : pos) - blockStart;
        std::streamoff i               = 0;

        while(i < blockSize)
        {
            size_t j         = i;
            const size_t end = std::min<size_t>(i + lineSize, blockSize);

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

void Bin2Asm(std::istream& source,
             std::ostream& targetHeader,
             std::ostream& targetAsm,
             std::ostream& targetBin,
             const fs::path& targetBinPath,
             const std::string& variable,
             bool nullTerminate,
             size_t bufferSize)
{
    assert(!variable.empty());

    source.seekg(0, std::ios::end);
    const auto sourceSize = source.tellg();
    source.seekg(0, std::ios::beg);

    const auto buffer = std::make_unique<char[]>(bufferSize);

    // Write header data
    targetHeader << "extern \"C\" const size_t " << variable << "_SIZE;" << std::endl;
    targetHeader << "extern \"C\" const char " << variable << "[" << sourceSize << "];"
                 << std::endl;

    const auto incbinOffset = targetBin.tellp();

    // Write binary data
    for(std::streamoff blockStart = 0; blockStart < sourceSize; blockStart += bufferSize)
    {
        source.read(buffer.get(), bufferSize);

        const auto pos       = source.tellg();
        const auto blockSize = (pos < 0 ? sourceSize : pos) - blockStart;

        targetBin.write(buffer.get(), blockSize);
    }

    if(nullTerminate)
        targetBin.put(0);

    // Write assembly
    const auto incbinSize  = targetBin.tellp() - incbinOffset;
    const auto writeSymbol = [&](const std::string& symbol, auto f) {
        targetAsm << "#if defined(_WIN32) || defined(__CYGWIN__)\n";
        targetAsm << "     /* PE/COFF format */\n";
        targetAsm << "     .section .rdata\n";
        targetAsm << "#else\n";
        targetAsm << "     /* ELF format */\n";
        targetAsm << "     .section .note.GNU-stack,\"\",@progbits\n";
        targetAsm << "     .type \"" << symbol << "\",@object\n";
        targetAsm << "     .section .rodata\n";
        targetAsm << "#endif\n";
        targetAsm << "    .globl \"" << symbol << "\"\n";
        targetAsm << "    .p2align 8\n";
        targetAsm << symbol << ":\n";
        f();
        targetAsm << "#if !(defined(_WIN32) || defined(__CYGWIN__))\n";
        targetAsm << "    .size " << symbol << ", . - " << symbol << "\n";
        targetAsm << "#endif\n";
    };

    writeSymbol(variable, [&] {
        targetAsm << "    .incbin \"" << targetBinPath.string() << "\", " << incbinOffset << ", "
                  << incbinSize << "\n";
    });

    writeSymbol(variable + "_SIZE", [&] { targetAsm << "    .quad " << incbinSize << "\n"; });
}

void PrintHelp()
{
    std::cout << "Usage: addkernels {<option>}" << std::endl;
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
    std::cout << "           -m[ark-includes] : mark variables that represent include files with "
                 "'_INCLUDE'. Default: off"
              << std::endl;
    std::cout << "           -a[sm] <asm> <bin> : inline source via assembly file and binary "
              << "instead of header. Default: off" << std::endl;
}

[[noreturn]] void WrongUsage(std::string_view error)
{
    std::cout << "Wrong usage: " << error << "\n" << std::endl;
    PrintHelp();
    // NOLINTNEXTLINE (concurrency-mt-unsafe)
    std::exit(1);
}

[[noreturn]] void UnknownArgument(const std::string_view arg)
{
    std::ostringstream ss;
    ss << "unknown argument - " << arg;
    WrongUsage(ss.str());
}

void Process(const fs::path& sourcePath,
             std::ostream& target,
             size_t bufferSize,
             size_t lineSize,
             bool recurse,
             bool as_extern,
             bool mark_includes,
             std::ofstream& asmStream,
             std::ofstream& binStream,
             const fs::path& targetBinPath)
{
    if(!fs::exists(sourcePath))
    {
        std::cerr << "File not found: " << sourcePath << std::endl;
        // NOLINTNEXTLINE (concurrency-mt-unsafe)
        std::exit(1);
    }

    fs::path root{sourcePath.has_parent_path() ? sourcePath.parent_path() : ""};
    std::ifstream sourceFile{sourcePath, std::ios::in | std::ios::binary};
    std::istream* source = &sourceFile;

    if(!sourceFile.is_open())
    {
        std::cerr << "Error opening file: " << sourcePath << std::endl;
        // NOLINTNEXTLINE (concurrency-mt-unsafe)
        std::exit(1);
    }

    const auto is_asm    = sourcePath.extension() == ".s";
    const auto is_cl     = sourcePath.extension() == ".cl";
    const auto is_hip    = sourcePath.extension() == ".cpp";
    const auto is_header = sourcePath.extension() == ".hpp";

    std::stringstream inlinerTemp;
    if(is_asm || is_cl || is_hip || is_header)
    {
        IncludeInliner inliner;
        try
        {
            if(is_asm)
            {
                inliner.Process(
                    sourceFile, inlinerTemp, root, sourcePath, ".include", false, recurse);
            }
            else if(is_cl || is_header)
            {
                inliner.Process(
                    sourceFile, inlinerTemp, root, sourcePath, "#include", true, recurse);
            }
            else if(is_hip)
            {
                inliner.Process(
                    sourceFile, inlinerTemp, root, sourcePath, "<#not_include>", true, false);
            }
        }
        catch(const InlineException& ex)
        {
            std::cerr << ex.What() << '\n' << ex.GetTrace() << std::endl;
            // NOLINTNEXTLINE (concurrency-mt-unsafe)
            std::exit(1);
        }
        source = &inlinerTemp;
    }

    auto variable{sourcePath.stem().string()};
    std::transform(variable.begin(), variable.end(), variable.begin(), ::toupper);

    if(mark_includes)
        variable = variable + "_INCLUDE";

    if(as_extern && variable.length() != 0)
    {
        variable = "MIOPEN_KERNEL_" + variable;
    }

    if(asmStream.is_open())
    {
        assert(binStream.is_open());
        Bin2Asm(*source, target, asmStream, binStream, targetBinPath, variable, true, bufferSize);
    }
    else
    {
        Bin2Hex(*source, target, variable, true, bufferSize, lineSize);
    }
}

int main(int argc, char* argv[])
{
    if(argc == 1)
    {
        PrintHelp();
        return 2;
    }

    // The configuration to establish with command line options
    // before running the algorithm.

    std::string guard;
    size_t bufferSize = 512;
    size_t lineSize   = 16;

    std::string targetFile;
    std::string asmOutputFile;
    std::string binOutputFile;
    std::vector<fs::path> sourceFiles;

    bool recurse       = true;
    bool as_extern     = false;
    bool mark_includes = false;

    // Parse command line options to establish configuration

    int i = 0;
    while(++i < argc)
    {
        std::string arg(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);

        if(arg == "-s" || arg == "-source")
        {
            while(++i < argc && *argv[i] != '-')
                sourceFiles.emplace_back(argv[i]);
        }
        else if(arg == "-t" || arg == "-target")
        {
            std::string outputFile{argv[++i]};
            if(!targetFile.empty())
                std::cerr << "Warning: overriding output file\n    '" << targetFile
                          << "'\nwith\n    '" << outputFile << "'\n";
            targetFile = outputFile;
        }
        else if(arg == "-l" || arg == "-line-size")
        {
            lineSize = std::stol(argv[++i]);
        }
        else if(arg == "-b" || arg == "-buffer")
        {
            bufferSize = std::stol(argv[++i]);
        }
        else if(arg == "-g" || arg == "-guard")
        {
            guard = argv[++i];
        }
        else if(arg == "-n" || arg == "-no-recurse")
        {
            recurse = false;
        }
        else if(arg == "-m" || arg == "-mark-includes")
        {
            mark_includes = true;
        }
        else if(arg == "-e" || arg == "-extern")
        {
            as_extern = true;
        }
        else if(arg == "-a" || arg == "-asm")
        {
            if(i + 2 >= argc)
            {
                std::ostringstream ss;
                ss << arg << " requires arguments <asm path> <bin path>";
                WrongUsage(ss.str());
            }

            std::string outputAsm{argv[++i]};
            if(!asmOutputFile.empty())
                std::cerr << "Warning: overriding asm output file\n    '" << asmOutputFile
                          << "'\nwith\n    '" << outputAsm << "'\n";

            asmOutputFile = outputAsm;

            std::string outputBin{argv[++i]};
            binOutputFile = outputBin;
        }
        else
        {
            UnknownArgument(arg);
        }
    }

    // Execute the algorithm on the established configuration

    if(sourceFiles.empty())
        WrongUsage("'source' option is required");

    std::stringstream ss;
    if(guard.length() > 0)
    {
        ss << "#ifndef " << guard << "\n#define " << guard << "\n";
    }

    ss << "#ifndef MIOPEN_USE_CLANG_TIDY\n"
          "#include <cstddef>\n";

    std::ofstream asmStream;
    std::ofstream binStream;
    if(!asmOutputFile.empty())
    {
        assert(!binOutputFile.empty());
        asmStream.open(asmOutputFile, std::ios::out | std::ios::binary);
        binStream.open(binOutputFile, std::ios::out | std::ios::binary);

        if(!asmStream.is_open())
        {
            std::cerr << "failure opening file: " << asmOutputFile << "\n";
            return 1;
        }

        if(!binStream.is_open())
        {
            std::cerr << "failure opening file: " << binOutputFile << "\n";
            return 1;
        }
    }

    for(const auto& file : sourceFiles)
    {
        Process(file,
                ss,
                bufferSize,
                lineSize,
                recurse,
                as_extern,
                mark_includes,
                asmStream,
                binStream,
                binOutputFile);
    }

    ss << "#endif\n";

    if(guard.length() > 0)
    {
        ss << "#endif\n";
    }

    auto sourceCode = ss.str();

    if(targetFile.empty())
    {
        std::cout << sourceCode << std::flush;
    }
    else
    {
        std::ofstream file{targetFile};
        if(!file.is_open())
        {
            std::cerr << "failure opening file: " << targetFile << "\n";
            return 1;
        }
        file.write(sourceCode.data(), sourceCode.length());
    }

    return 0;
}
