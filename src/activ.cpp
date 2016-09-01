#include<mlopen/activ.hpp>
#include <cassert>

namespace mlopen {

ActivationDescriptor::ActivationDescriptor() {}

ActivationDescriptor::ActivationDescriptor(mlopenActivationMode_t m, const double *pparms) 
: parms(pparms, pparms+3), mode(m) {}

mlopenActivationMode_t ActivationDescriptor::GetMode() const
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
}
