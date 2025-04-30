#ifndef REDUCE_TUNABLES_HPP
#define REDUCE_TUNABLES_HPP

struct tunable_generic_reduction
{
    int BlockSize;
    int GredThreadBufferLength;
    int GredAccessesPerThreadInBlock;
    int GredAccessesPerThreadInWarp;
};

static const struct tunable_generic_reduction default_tunable_generic_reduction = {256, 8, 2, 2};

#endif
