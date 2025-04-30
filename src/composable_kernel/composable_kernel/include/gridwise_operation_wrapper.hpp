#ifndef CK_GRIDWISE_OPERATION_KERNEL_WRAPPER
#define CK_GRIDWISE_OPERATION_KERNEL_WRAPPER

template <typename GridwiseOp, typename... Xs>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        run_gridwise_operation(Xs... xs)
{
    GridwiseOp{}.Run(xs...);
}

#endif
