#include <mlopen/context.hpp>
#include <mlopen/errors.hpp>
#include <cstdio>

extern "C" 
mlopenStatus_t mlopenCreate(mlopenHandle_t *handle,
		int numStreams,
		mlopenAcceleratorQueue_t *streams ) {

	return mlopen::try_([&] {
		if(numStreams != 0) {
			mlopen::deref(handle) = new mlopen::Context(numStreams, streams);
		}
		else {
			mlopen::deref(handle) = new mlopen::Context();
		}
	});
}

// TODO: Stream size should be a spearate parameter
extern "C"
mlopenStatus_t mlopenGetStream(mlopenHandle_t handle,
		mlopenAcceleratorQueue_t *streamID,
		int numStream) {
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
