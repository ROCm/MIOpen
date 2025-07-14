#ifndef CK_MULTI_INDEX_HPP
#define CK_MULTI_INDEX_HPP

#include "common_header.hpp"

#if CK_USE_DYNAMICALLY_INDEXED_MULTI_INDEX
#include "array_multi_index.hpp"
#else
#include "statically_indexed_array_multi_index.hpp"
#endif

#endif
