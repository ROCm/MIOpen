/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_OLC_TARGET_PROPERTIES_HPP
#define GUARD_OLC_TARGET_PROPERTIES_HPP

#include <boost/optional.hpp>
#include <string>

namespace olCompile {

struct Handle;

struct TargetProperties
{
    const std::string& Name() const { return name; }
    const std::string& DbId() const { return dbId; }
    boost::optional<bool> Xnack() const { return xnack; }
    boost::optional<bool> Sramecc() const { return sramecc; }
    boost::optional<bool> SrameccReported() const { return sramecc_reported; }
    void Init(const Handle*);

    private:
    void InitDbId();
    std::string name;
    std::string dbId;
    boost::optional<bool> xnack            = boost::none;
    boost::optional<bool> sramecc          = boost::none;
    boost::optional<bool> sramecc_reported = boost::none;
};

} // namespace olCompile

#endif // GUARD_OLC_TARGET_PROPERTIES_HPP
