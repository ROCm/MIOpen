// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef CK_ENABLE_IF_HPP
#define CK_ENABLE_IF_HPP

#include "miopen_type_traits.hpp"

namespace ck {

template <bool B, typename T = void>
using enable_if = std::enable_if<B, T>;

template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

} // namespace ck
#endif
