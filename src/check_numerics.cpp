#include <miopen/check_numerics.hpp>
#include <miopen/env.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_CHECK_NUMERICS)

int CheckNumericsEnabled(int bitMask) { return (miopen::Value(MIOPEN_CHECK_NUMERICS{})) & bitMask; }

// Must keep this structure synchronized with one in MIOpenCheckNumerics
struct CheckNumericsResult
{
    float sum    = 0.0f;
    float absSum = 0.0f;
    float min    = 0.0f;
    float max    = 0.0f;

    int hasZero = 0;
    int hasNan  = 0;
    int hasInf  = 0;
};

bool checkNumericsImpl(
    Handle& handle, int mode, const TensorDescriptor& dDesc, ConstData_t data, bool isInput)
{
    int numElements = dDesc.GetElementSize();

    // TODO - some constants we should get from the device:
    const int blockSize             = 256;
    const int numBlocks             = handle.GetMaxComputeUnits() * 6;
    const size_t numGlobalWorkItems = blockSize * numBlocks;

    const int computeStats = (mode & CheckNumerics::ComputeStats);

    CheckNumericsResult abnormal_h;

    auto abnormal_d =
        handle.Create(sizeof(CheckNumericsResult)); // TODO - someday avoid slow malloc/free here
    handle.WriteTo(&abnormal_h, abnormal_d, sizeof(CheckNumericsResult));

    std::string program_name      = "MIOpenCheckNumerics.cl";
    std::string kernel_name       = "MIOpenCheckNumerics";
    const std::vector<size_t> vld = {size_t{blockSize}, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {numGlobalWorkItems, size_t{1}, size_t{1}};
    handle.GetKernel("MIOpenCheckNumerics", "", program_name, kernel_name, vld, vgd, "")(
        data, numElements, abnormal_d.get(), computeStats);

    handle.ReadTo(&abnormal_h, abnormal_d, sizeof(CheckNumericsResult));

    bool isAbnormal = abnormal_h.hasNan || abnormal_h.hasInf;

    if((mode & CheckNumerics::Info) || ((mode & CheckNumerics::Warn) && isAbnormal))
    {

        std::cerr << (isAbnormal ? "warn:" : "info:") << " checkNumerics on"
                  << " " << (isInput ? "INPUT " : "OUTPUT") << " ptr=" << data
                  << " zeros=" << abnormal_h.hasZero << " nans=" << abnormal_h.hasNan
                  << " infs=" << abnormal_h.hasInf;
        if(computeStats)
        {
            std::cerr << " mean=" << abnormal_h.sum / numElements
                      << " absmean=" << abnormal_h.absSum / numElements << " min=" << abnormal_h.min
                      << " max=" << abnormal_h.max;
        }
        std::cerr << "  {" << dDesc << "}"
                  << "\n";
    }

    if(isAbnormal)
    {

        if((mode & CheckNumerics::Throw))
        {
            if(isInput)
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "abnormal checkNumerics result detected on INPUT");
            }
            else
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "abnormal checkNumerics result detected on OUTPUT");
            }
        }
        if((mode & CheckNumerics::Abort))
        {
            abort();
        }
    }

    return isAbnormal;
};

// Checks data for input
// Returns: 1 if abnormal value (inf or nan) detected in specified data, 0 otherwise
bool checkNumericsInput(Handle& handle, const TensorDescriptor& dDesc, ConstData_t data)
{
    return checkNumericsImpl(handle, miopen::Value(MIOPEN_CHECK_NUMERICS{}), dDesc, data, true);
}

// Synchronizes to wait for kernel to finish, then checks data for output:
// Returns: 1 if abnormal value (inf or nan) detected in specified data, 0 otherwise
bool checkNumericsOutput(Handle& handle, const TensorDescriptor& dDesc, Data_t data)
{
    handle.Finish();

    return checkNumericsImpl(handle, miopen::Value(MIOPEN_CHECK_NUMERICS{}), dDesc, data, false);
}

} // namespace miopen
