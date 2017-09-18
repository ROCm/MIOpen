#include <stdint.h>
#include <miopen/check_numerics.hpp>
#include <miopen/env.hpp>

namespace miopen {

MIOPEN_DECLARE_ENV_VAR(MIOPEN_CHECK_NUMERICS)

namespace CheckNumerics
{
    static const int Info                = 0x01;   // print results from all checks
    static const int Warn                = 0x02;   // print only if abnormal detected
    static const int Throw               = 0x04;   // MIOPEN_THROW on abnormal result
    static const int Abort               = 0x08;   // abort on abnormal result (to drop into debugger)
    static const int ComputeFloatStats   = 0x10;  // Print mean/absmean/min/max (slow)
}
int CheckNumericsEnabled(int bitMask) { return (miopen::Value(MIOPEN_CHECK_NUMERICS{})) & bitMask; }

#define DTYPE     float
#define ACCUMTYPE float 
// Must keep this structure synchronized with one in MIOpenCheckNumerics
struct CheckNumericsResult 
{
    ACCUMTYPE _sum = 0.0f;
    ACCUMTYPE _absSum = 0.0f;
    DTYPE     _min = 0.0f;
    DTYPE     _max = 0.0f;

    int       _hasZero = 0;
    int       _hasNan = 0;
    int       _hasInf = 0;
};


static bool checkNumericsImpl(Handle &handle, const TensorDescriptor &dDesc, ConstData_t data, bool isInput) 
{
    int numElements = dDesc.GetElementSize();

    // TODO - some constants we should get from the device:
    const int blockSize = 256;
    const int numBlocks = handle.GetMaxComputeUnits() * 6;
    const size_t numGlobalWorkItems = blockSize * numBlocks;

    const int computeFloatStats = CheckNumericsEnabled(CheckNumerics::ComputeFloatStats);

    CheckNumericsResult abnormal_h;

    auto abnormal_d = handle.Create(sizeof(CheckNumericsResult)); // TODO - someday avoid slow malloc/free here
    handle.WriteTo(&abnormal_h, abnormal_d, sizeof(CheckNumericsResult)); 

    std::string program_name = "MIOpenCheckNumerics.cl";
    std::string kernel_name  = "MIOpenCheckNumerics";
    const std::vector<size_t> vld = {size_t{blockSize}, size_t{1}, size_t{1}};
    const std::vector<size_t> vgd = {numGlobalWorkItems, size_t{1}, size_t{1}};
    handle.GetKernel("MIOpenCheckNumerics", "", program_name, kernel_name, vld, vgd, "" ) (data, numElements, abnormal_d.get(), computeFloatStats); 

    handle.ReadTo(&abnormal_h, abnormal_d, sizeof(CheckNumericsResult));

    bool isAbnormal = abnormal_h._hasNan || abnormal_h._hasInf;

    if ( CheckNumericsEnabled(CheckNumerics::Info) ||
        (CheckNumericsEnabled(CheckNumerics::Warn) && isAbnormal)) {


        std::cerr << (isAbnormal ? "warn:" : "info:")
                  << " checkNumerics on" 
                  << " " << (isInput ? "INPUT ":"OUTPUT")
                  << " ptr=" << data 
                  << " zeros=" << abnormal_h._hasZero
                  << " nans="  << abnormal_h._hasNan
                  << " infs="  << abnormal_h._hasInf;
        if (computeFloatStats) {
            std::cerr << " mean="  << abnormal_h._sum / numElements
                      << " absmean="  << abnormal_h._absSum / numElements
                      << " min="  << abnormal_h._min 
                      << " max="  << abnormal_h._max ;
        }
        std::cerr << "  {" << dDesc  << "}"
                  << "\n";
    }

    if (isAbnormal) {

        if (CheckNumericsEnabled(CheckNumerics::Throw)) {
            MIOPEN_THROW(miopenStatusInternalError, "abnormal checkNumerics result detected");
        }
        if (CheckNumericsEnabled(CheckNumerics::Abort)) {
            abort();
        }
    }


    return isAbnormal;
};


// Checks data for input
// Returns: 1 if abnormal value (inf or nan) detected in specified data, 0 otherwise
bool checkNumericsInput(Handle &handle, const TensorDescriptor &dDesc, ConstData_t data) 
{
    return checkNumericsImpl(handle, dDesc, data, true);
}


// Synchronizes to wait for kernel to finish, then checks data for output:
// Returns: 1 if abnormal value (inf or nan) detected in specified data, 0 otherwise
bool checkNumericsOutput(Handle &handle, const TensorDescriptor &dDesc, Data_t data) 
{
    handle.Finish(); 

    return checkNumericsImpl(handle, dDesc, data, false);
}

} // namespace miopen
