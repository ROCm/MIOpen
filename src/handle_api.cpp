#include <mlopen/handle.hpp>
#include <mlopen/errors.hpp>
#include <cstdio>

extern "C" 
mlopenStatus_t mlopenCreate(mlopenHandle_t *handle) {

	return mlopen::try_([&] {
			mlopen::deref(handle) = new mlopen::Handle();
	});
}

extern "C" 
mlopenStatus_t mlopenCreateWithStream(mlopenHandle_t *handle,
		mlopenAcceleratorQueue_t *stream ) {

	return mlopen::try_([&] {
			mlopen::deref(handle) = new mlopen::Handle(stream);
	});
}

extern "C"
mlopenStatus_t mlopenGetStream(mlopenHandle_t handle,
		mlopenAcceleratorQueue_t *streamID) {
	return mlopen::try_([&] {
		mlopen::deref(streamID) = mlopen::deref(handle).GetStream();
	});
}

extern "C"
mlopenStatus_t mlopenDestroy(mlopenHandle_t handle) {
	return mlopen::try_([&] {
		mlopen_destroy_object(handle);
	});
}

extern "C" mlopenStatus_t mlopenGetKernelTime(mlopenHandle_t handle, float* time)
{
	return mlopen::try_([&] {
		mlopen::deref(time) = mlopen::deref(handle).GetKernelTime();
	});
}
extern "C" mlopenStatus_t mlopenEnableProfiling(mlopenHandle_t handle, bool enable)
{
	return mlopen::try_([&] {
		mlopen::deref(handle).EnableProfiling(enable);
	});
}
