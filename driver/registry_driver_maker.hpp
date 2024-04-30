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
#ifndef GUARD_DRIVER_REGISTRY_DRIVER_MAKER_HPP
#define GUARD_DRIVER_REGISTRY_DRIVER_MAKER_HPP

#include "driver.hpp"

#include <string>
#include <vector>

namespace rdm {

/// The function of the DriverMaker type should behave as follows:
/// If \p base_arg matches the command-line driver name, then
/// instantiates the driver object and returns pointer to it.
/// Otherwise, returns nullptr.
using DriverMaker = Driver* (*)(const std::string& base_arg);

const std::vector<DriverMaker>& GetRegistry();

namespace impl {
bool Register(DriverMaker f);
} // namespace impl

} // namespace rdm

/// Registers the function of the DriverMaker type.
#define REGISTER_DRIVER_MAKER(name) static bool CALL_ONCE##name = ::rdm::impl::Register(name)

#endif // GUARD_DRIVER_REGISTRY_DRIVER_MAKER_HPP
