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
/**********************************************************************
Copyright (c)2016 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef MLO_UITLS_H_
#define MLO_UITLS_H_

#include <miopen/errors.hpp>
#include <miopen/manage_ptr.hpp>
#include <map>
#include <fstream>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
// #include <BaseTsd.h>
#include <direct.h>
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
//#ifndef getcwd
// #define getcwd _getcwd
//#endif
typedef unsigned int uint;

#ifndef getcwd
#define getcwd _getcwd
#endif

#else // !WIN32 so Linux and APPLE
#include <climits>
#include <unistd.h>
#include <cstdbool>
#include <sys/time.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif
using __int64 = long long;
#endif

using mlo_ocl_arg     = std::pair<size_t, void*>;
using mlo_ocl_args    = std::map<int, mlo_ocl_arg>;
using manage_file_ptr = MIOPEN_MANAGE_PTR(FILE*, fclose);

#ifdef MIOPEN

#ifdef WIN32

inline double miopen_mach_absolute_time(void) // Windows
{
    double ret = 0;
    __int64 frec;
    __int64 clocks;
    QueryPerformanceFrequency((LARGE_INTEGER*)&frec);
    QueryPerformanceCounter((LARGE_INTEGER*)&clocks);
    ret = (double)clocks * 1000. / (double)frec;
    return (ret);
}
#else
// We want milliseconds. Following code was interpreted from timer.cpp
inline double miopen_mach_absolute_time() // Linux
{
    double d = 0.0;
    timeval t{};
    t.tv_sec  = 0;
    t.tv_usec = 0;
    gettimeofday(&t, nullptr);
    d = (t.tv_sec * 1000.0) + t.tv_usec / 1000.0; // TT: was 1000000.0
    return (d);
}
#endif

inline double subtractTimes(double endTime, double startTime)
{
    double difference        = endTime - startTime;
    static double conversion = 0.0;

    if(conversion == 0.0)
    {
#ifdef __APPLE__
        mach_timebase_info_data_t info{};
        kern_return_t err = mach_timebase_info(&info);

        // Convert the timebase into seconds
        if(err == 0)
        {
            conversion = 1e-9 * static_cast<double>(info.numer) / static_cast<double>(info.denom);
        }
#else
        conversion = 1.;
#endif
    }
    return conversion * difference;
}

#endif

/**
 * for the opencl program file processing
 */
class mloFile
{
    public:
    /**
     *Default constructor
     */
    mloFile() : source_("") {}

    /**
     * Destructor
     */
    ~mloFile() = default;

    /**
     * Opens the CL program file
     * @return true if success else false
     */
    bool open(const char* fileName)
    {
        std::vector<char> str;
        // Open file stream
        std::fstream f(fileName, (std::fstream::in | std::fstream::binary));
        // Check if we have opened file stream
        if(f.is_open())
        {
            size_t sizeFile;
            // Find the stream size
            f.seekg(0, std::fstream::end);
            size_t size = sizeFile = static_cast<size_t>(f.tellg());
            f.seekg(0, std::fstream::beg);
            str = std::vector<char>(size + 1);
            // Read file
            f.read(str.data(), sizeFile);
            f.close();
            str[size] = '\0';
            source_   = str.data();
            return true;
        }
        return false;
    }

    /**
     * writeBinaryToFile
     * @param fileName Name of the file
     * @param binary char binary array
     * @param numBytes number of bytes
     * @return true if success else false
     */
    int writeBinaryToFile(const char* fileName, const char* binary, size_t numBytes)
    {
        manage_file_ptr output{fopen(fileName, "wb")};

        if(output == nullptr)
        {
            return 0;
        }
        fwrite(binary, sizeof(char), numBytes, output.get());
        return 0;
    }

    /**
     * readBinaryToFile
     * @param fileName name of file
     * @return true if success else false
     */
    int readBinaryFromFile(const char* fileName)
    {
        manage_file_ptr input{fopen(fileName, "rb")};
        if(input == nullptr)
        {
            // TODO: Should throw
            return -1;
            // MIOPEN_THROW("Error opening file " + std::string(fileName));;
        }
        fseek(input.get(), 0L, SEEK_END);
        auto size = ftell(input.get());
        rewind(input.get());
        std::vector<char> binary(size);
        auto val = fread(binary.data(), sizeof(char), size, input.get());
        if(val != size)
            MIOPEN_THROW("Error reading file");
        source_.assign(binary.data(), size);
        return 0;
    }

    /**
     * Replaces Newline with spaces
     */
    void replaceNewlineWithSpaces()
    {
        size_t pos = source_.find_first_of('\n', 0);
        while(pos != -1)
        {
            source_.replace(pos, 1, " ");
            pos = source_.find_first_of('\n', pos + 1);
        }
        pos = source_.find_first_of('\r', 0);
        while(pos != -1)
        {
            source_.replace(pos, 1, " ");
            pos = source_.find_first_of('\r', pos + 1);
        }
    }

    /**
     * source
     * Returns a pointer to the string object with the source code
     */
    const std::string& source() const { return source_; }

    /**
     * Disable copy constructor
     */
    mloFile(const mloFile&) = delete;

    /**
     * Disable operator=
     */
    mloFile& operator=(const mloFile&) = delete;

    private:
    std::string source_; //!< source code of the CL program
};

inline void tokenize(const std::string& str,
                     std::vector<std::string>& tokens,
                     const std::string& delimiters = " ");

inline void
tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters)
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);

    while(std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}
#endif
