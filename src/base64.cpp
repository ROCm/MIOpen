/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <miopen/errors.hpp>
#include <miopen/utility/base64.hpp>

namespace miopen {

namespace {

const std::string_view base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                     "abcdefghijklmnopqrstuvwxyz"
                                     "0123456789+/";

} // namespace

std::string base64Encode(const uint8_t* data, std::size_t length)
{
    // Calculate the exact size of the resulting Base64 string
    size_t outputSize = ((length + 2) / 3) * 4;
    std::string encodedString;
    encodedString.reserve(outputSize); // Preallocate memory for the result

    size_t i = 0;
    unsigned char charArray3[3];
    unsigned char charArray4[4];

    while(length-- != 0U)
    {
        charArray3[i++] = *(data++);
        if(i == 3)
        {
            charArray4[0] = (charArray3[0] & 0xfc) >> 2;
            charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
            charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
            charArray4[3] = charArray3[2] & 0x3f;

            std::transform(std::begin(charArray4),
                           std::end(charArray4),
                           std::back_inserter(encodedString),
                           [](unsigned char c) { return base64Chars[c]; });
            i = 0;
        }
    }

    if(i != 0U)
    {
        for(size_t j = i; j < 3; j++)
        {
            charArray3[j] = '\0';
        }

        charArray4[0] = (charArray3[0] & 0xfc) >> 2;
        charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
        charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
        charArray4[3] = charArray3[2] & 0x3f;

        for(size_t j = 0; j < i + 1; j++)
        {
            encodedString += base64Chars[charArray4[j]];
        }

        while(i++ < 3)
        {
            encodedString += '=';
        }
    }

    return encodedString;
}

std::vector<uint8_t> base64Decode(const std::string_view& encodedString)
{
    size_t length = encodedString.size();
    MIOPEN_THROW_IF(length % 4 != 0, "Invalid Base64 input");

    // Reserve space to avoid reallocations
    size_t outputLength = (length / 4) * 3;
    if(encodedString[length - 1] == '=')
        outputLength--;
    if(encodedString[length - 2] == '=')
        outputLength--;

    std::vector<uint8_t> decodedData;
    decodedData.reserve(outputLength);

    uint8_t charArray3[3];
    uint8_t charArray4[4];
    size_t i = 0;

    for(char c : encodedString)
    {
        if(c == '=')
            break;

        auto it = std::find(base64Chars.begin(), base64Chars.end(), c);
        MIOPEN_THROW_IF(it == base64Chars.end(), "Invalid character in Base64 string");

        charArray4[i++] = static_cast<uint8_t>(std::distance(base64Chars.begin(), it));
        if(i == 4)
        {
            charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
            charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
            charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

            decodedData.insert(decodedData.end(), charArray3, charArray3 + 3);
            i = 0;
        }
    }

    if(i != 0U)
    {
        for(size_t j = i; j < 4; j++)
        {
            charArray4[j] = 0;
        }

        charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
        charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
        charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

        for(size_t j = 0; j < i - 1; j++)
        {
            decodedData.push_back(charArray3[j]);
        }
    }

    return decodedData;
}

} // namespace miopen
