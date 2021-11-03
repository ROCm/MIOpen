#ifndef DEBUG_HPP
#define DEBUG_HPP

namespace debug {
namespace debug_driver_gemm_xdlops_v2r3 {

// these vars are on host, they control block_id to C matrix tile idx (m0, n0) mapping
static ck::index_t M01 = 1;
static ck::index_t N01 = 1;

} // namespace debug_driver_gemm_xdlops_v2r3
} // namespace debug
#endif
