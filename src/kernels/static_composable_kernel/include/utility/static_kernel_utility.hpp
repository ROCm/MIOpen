#ifndef CK_UTILITY_HPP
#define CK_UTILITY_HPP

#include "static_kernel_config.hpp"

namespace ck {

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

} // namspace ck

#endif
