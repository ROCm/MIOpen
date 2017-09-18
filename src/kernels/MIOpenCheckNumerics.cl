// Must keep this structure synchronized with one in MIOpenCheckNumerics
struct CheckNumericsResult 
{
    int _hasZero;
    int _hasNan;
    int _hasInf;
};

#define DTYPE     float
#define ACCUMTYPE float 
// Checks a block of data for abnormal numeric values :
__kernel void MIOpenCheckNumerics(const __global DTYPE *data, int size, __global struct CheckNumericsResult *abnormal)
{
    const int gid           = get_global_id(0);
    const int total_wi_size = get_global_size(0);

    int offset = gid;
    ACCUMTYPE acc = 0.0f;
    while (offset < size) {
        DTYPE value = data[offset];
        acc += value;
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

    //atomic_add(&abnormal->_sum, acc);
}
