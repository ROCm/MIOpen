#include <miopen/handle.hpp>
#include <miopen/errors.hpp>
#include <cstdio>

extern "C" 
miopenStatus_t miopenCreate(miopenHandle_t *handle) {

	return miopen::try_([&] {
			miopen::deref(handle) = new miopen::Handle();
	});
}

extern "C" 
miopenStatus_t miopenCreateWithStream(miopenHandle_t *handle,
		miopenAcceleratorQueue_t *stream ) {

	return miopen::try_([&] {
			miopen::deref(handle) = new miopen::Handle(stream);
	});
}

extern "C"
miopenStatus_t miopenGetStream(miopenHandle_t handle,
		miopenAcceleratorQueue_t *streamID) {
	return miopen::try_([&] {
		miopen::deref(streamID) = miopen::deref(handle).GetStream();
	});
}

extern "C"
miopenStatus_t miopenDestroy(miopenHandle_t handle) {
	return miopen::try_([&] {
		miopen_destroy_object(handle);
	});
}

extern "C" miopenStatus_t miopenGetKernelTime(miopenHandle_t handle, float* time)
{
	return miopen::try_([&] {
		miopen::deref(time) = miopen::deref(handle).GetKernelTime();
	});
}
extern "C" miopenStatus_t miopenEnableProfiling(miopenHandle_t handle, bool enable)
{
	return miopen::try_([&] {
		miopen::deref(handle).EnableProfiling(enable);
	});
}
