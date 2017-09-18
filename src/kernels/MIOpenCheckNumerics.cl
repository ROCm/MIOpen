#define DTYPE     float
#define ACCUMTYPE float 

// Must keep this structure synchronized with one in MIOpenCheckNumerics
struct CheckNumericsResult 
{
    ACCUMTYPE _sum;
    ACCUMTYPE _absSum;
    DTYPE     _min;
    DTYPE     _max;

    int       _hasZero;
    int       _hasNan;
    int       _hasInf;
};

union AtomicFloat {
   unsigned int u32;
   float        f32;
};

cl_atomic_add_float (volatile __global float *addr, float val)
{
    union AtomicFloat current, expected, next;

    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = current.f32 + val;

        current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                                    expected.u32, next.u32);   
    } while (current.u32 != expected.u32);
}

cl_atomic_add_float_local (volatile __local float *addr, float val)
{
    union AtomicFloat current, expected, next;

    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = current.f32 + val;

        current.u32  = atomic_cmpxchg( (volatile __local unsigned int *)addr, 
                                    expected.u32, next.u32);   
    } while (current.u32 != expected.u32);
}

cl_atomic_min_float (volatile __global float *addr, float val)
{
    union AtomicFloat current, expected, next;

    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = fmin(current.f32,val);

        current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                                    expected.u32, next.u32);   
    } while (current.u32 != expected.u32);
}


cl_atomic_max_float (volatile __global float *addr, float val)
{
    union AtomicFloat current, expected, next;

    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = fmax(current.f32,val);

        current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                                    expected.u32, next.u32);   
    } while (current.u32 != expected.u32);
}

// Checks a block of data for abnormal numeric values :
__kernel void MIOpenCheckNumerics(const __global DTYPE *data, int size, __global struct CheckNumericsResult *abnormal, int floatStats)
{
    const int gid           = get_global_id(0);
    const int total_wi_size = get_global_size(0);

    int offset = gid;
    ACCUMTYPE sum = 0.0f;
    ACCUMTYPE abssum = 0.0f;
    DTYPE minV = FLT_MAX;
    DTYPE maxV = FLT_MIN;
    while (offset < size) {
        DTYPE value = data[offset];
        sum += value;
        abssum += fabs(value);
        minV = min(minV, value);
        maxV = max(maxV, value);
        
        if (fabs(value) <= 0.0f) { // iszero check
           abnormal->_hasZero = 1;
        }
        if (isnan(value)) {
            abnormal->_hasNan  = 1;
        }
        if (isinf(value)) {
            abnormal->_hasInf  = 1;
        }
        offset += total_wi_size;
    }

    if (floatStats) {
        cl_atomic_add_float(&abnormal->_sum, sum);
        cl_atomic_add_float(&abnormal->_absSum, abssum);
        cl_atomic_min_float(&abnormal->_min, minV);
        cl_atomic_max_float(&abnormal->_max, maxV);
    }
}
