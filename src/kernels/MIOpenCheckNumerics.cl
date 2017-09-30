#define DTYPE float
#define ACCUMTYPE float

// Must keep this structure synchronized with one in MIOpenCheckNumerics
struct CheckNumericsResult
{
    float sum;
    float absSum;
    float min;
    float max;

    int hasZero;
    int hasNan;
    int hasInf;
};

union AtomicFloat
{
    unsigned int u32;
    float f32;
};

void cl_atomic_add_float(volatile __global float* addr, float val)
{
    union AtomicFloat current, expected, next;

    current.f32 = *addr;
    do
    {
        expected.f32 = current.f32;
        next.f32     = current.f32 + val;

        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while(current.u32 != expected.u32);
}

void cl_atomic_min_float(volatile __global float* addr, float val)
{
    union AtomicFloat current, expected, next;

    current.f32 = *addr;
    do
    {
        expected.f32 = current.f32;
        next.f32     = fmin(current.f32, val);

        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while(current.u32 != expected.u32);
}

void cl_atomic_max_float(volatile __global float* addr, float val)
{
    union AtomicFloat current, expected, next;

    current.f32 = *addr;
    do
    {
        expected.f32 = current.f32;
        next.f32     = fmax(current.f32, val);

        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while(current.u32 != expected.u32);
}

#define GROUP_SIZE 256
#define NUM_STATS 4

#define REDUCE_OPS(w)                                                             \
    if(lid < w)                                                                   \
    {                                                                             \
        stats[NUM_STATS * (lid) + 0] += stats[NUM_STATS * (lid + w) + 0];         \
        stats[NUM_STATS * (lid) + 1] += stats[NUM_STATS * (lid + w) + 1];         \
        stats[NUM_STATS * (lid) + 2] =                                            \
            fmin(stats[NUM_STATS * (lid) + 2], stats[NUM_STATS * (lid + w) + 2]); \
        stats[NUM_STATS * (lid) + 3] =                                            \
            fmax(stats[NUM_STATS * (lid) + 3], stats[NUM_STATS * (lid + w) + 3]); \
        barrier(CLK_LOCAL_MEM_FENCE);                                             \
    }

// Checks a block of data for abnormal numeric values :
__kernel void MIOpenCheckNumerics(const __global DTYPE* data,
                                  int size,
                                  __global struct CheckNumericsResult* abnormal,
                                  int computeStats)
{
    const int lid           = get_local_id(0);
    const int gid           = get_global_id(0);
    const int total_wi_size = get_global_size(0);

    local float stats[4 * GROUP_SIZE];

    int offset       = gid;
    ACCUMTYPE sum    = 0.0f;
    ACCUMTYPE abssum = 0.0f;
    DTYPE minV       = FLT_MAX;
    DTYPE maxV       = FLT_MIN;
    while(offset < size)
    {
        DTYPE value = data[offset];
        sum += value;
        abssum += fabs(value);
        minV = min(minV, value);
        maxV = max(maxV, value);

        if(fabs(value) <= 0.0f)
        { // iszero check
            abnormal->hasZero = 1;
        }
        if(isnan(value))
        {
            abnormal->hasNan = 1;
        }
        if(isinf(value))
        {
            abnormal->hasInf = 1;
        }
        offset += total_wi_size;
    }

    if(computeStats)
    {
        stats[NUM_STATS * lid + 0] = sum;
        stats[NUM_STATS * lid + 1] = abssum;
        stats[NUM_STATS * lid + 2] = minV;
        stats[NUM_STATS * lid + 3] = maxV;
        barrier(CLK_LOCAL_MEM_FENCE);

        REDUCE_OPS(128)
        REDUCE_OPS(64)
        REDUCE_OPS(32)
        REDUCE_OPS(16)
        REDUCE_OPS(8)
        REDUCE_OPS(4)
        REDUCE_OPS(2)
        REDUCE_OPS(1)

        if(lid == 0)
        {
            cl_atomic_add_float(&abnormal->sum, stats[0]);
            cl_atomic_add_float(&abnormal->absSum, stats[1]);
            cl_atomic_min_float(&abnormal->min, stats[2]);
            cl_atomic_max_float(&abnormal->max, stats[3]);
        }
    }
}
