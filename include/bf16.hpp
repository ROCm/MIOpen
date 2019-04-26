/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef BF16_H_
#define BF16_H_
#include <boost/operators.hpp>

class bfloat16 : boost::totally_ordered<bfloat16, boost::arithmetic<bfloat16>>
{
    public:
    bfloat16() : data_() {}
    explicit bfloat16(float rhs)
    {
        static union
        {
            std::uint32_t bf16_st;
            float float_st;
        } bits_st;

        bits_st.float_st = rhs;

        data_ = bits_st.bf16_st >> 16;
    }
    operator float() const
    {
        static union
        {
            std::uint32_t bf16_st;
            float float_st;
        } bits_st;

        bits_st.bf16_st = data_;
        bits_st.bf16_st = bits_st.bf16_st << 16;

        return bits_st.float_st;
    }

    bfloat16 operator-() const { return bfloat16(-static_cast<float>(*this)); }
    bfloat16 operator+() const { return *this; }

    bfloat16& operator=(const float rhs)
    {
        *this = bfloat16(rhs);
        return *this;
    }
    bfloat16& operator+=(bfloat16 rhs)
    {
        *this = bfloat16(static_cast<float>(*this) + static_cast<float>(rhs));
        return *this;
    }
    bfloat16& operator-=(bfloat16 rhs)
    {
        *this += -rhs;
        return *this;
    }
    bfloat16& operator*=(bfloat16 rhs)
    {
        *this = bfloat16(static_cast<float>(*this) * static_cast<float>(rhs));
        return *this;
    }
    bfloat16& operator/=(bfloat16 rhs)
    {
        *this = bfloat16(static_cast<float>(*this) / static_cast<float>(rhs));
        return *this;
    }

    bool operator<(bfloat16 rhs) const
    {
        return static_cast<float>(*this) < static_cast<float>(rhs);
    }
    bool operator==(bfloat16 rhs) const { return std::equal_to<float>()(*this, rhs); }
    private:
    std::uint16_t data_;
};

#endif
