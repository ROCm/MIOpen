#include <miopen/activ.hpp>
#include <miopen/logger.hpp>
#include <cassert>

namespace miopen {

ActivationDescriptor::ActivationDescriptor() {}

ActivationDescriptor::ActivationDescriptor(miopenActivationMode_t m, const double *pparms) 
: parms(pparms, pparms+3), mode(m) {}

ActivationDescriptor::ActivationDescriptor(miopenActivationMode_t m, double alpha, double beta, double power) 
: parms({alpha, beta, power}), mode(m) 
{}

miopenActivationMode_t ActivationDescriptor::GetMode() const
{
	return this->mode;
}

double ActivationDescriptor::GetAlpha() const
{
	return this->parms[0];
}

double ActivationDescriptor::GetBeta() const
{
	return this->parms[1];
}

double ActivationDescriptor::GetPower() const
{
	return this->parms[2];
}
std::ostream& operator<< (std::ostream& stream, const ActivationDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream, x.mode,
        miopenActivationPATHTRU,
        miopenActivationLOGISTIC,
        miopenActivationTANH,
        miopenActivationRELU,
        miopenActivationSOFTRELU,
        miopenActivationABS,
        miopenActivationPOWER
        // miopenActivationBRELU,
        // miopenActivationSQUARE,
        // miopenActivationSQR,
        // miopenActivationLINEAR
    ) << ", ";
    LogRange(stream, x.parms, ", ") << ", ";
    return stream;
}
}  // namespace miopen
