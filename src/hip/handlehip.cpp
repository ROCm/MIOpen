#include <mlopen/handle.hpp>

namespace mlopen {

struct HandleImpl
{
};

Handle::Handle (int numStreams, mlopenAcceleratorQueue_t *streams) 
: impl(new HandleImpl())
{
}

Handle::Handle () 
: impl(new HandleImpl())
{}

Handle::~Handle() {}

mlopenAcceleratorQueue_t Handle::GetStream() const
{

}

void Handle::EnableProfiling(bool enable)
{

}

float Handle::GetKernelTime() const
{
    return 0.0;
}

ManageDataPtr Handle::Create(int sz)
{

}
ManageDataPtr& Handle::WriteTo(const void* data, ManageDataPtr& ddata, int sz)
{

}
void Handle::ReadTo(void* data, const ManageDataPtr& ddata, int sz)
{

}
}

