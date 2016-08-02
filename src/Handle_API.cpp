#include "Handle.hpp"
#include <errors.hpp>
#include <cstdio>

extern "C" 
mlopenStatus_t mlopenCreate(mlopenHandle_t *handle,
		int numStreams,
		mlopenAcceleratorQueue_t *streams ) {

	return mlopen::try_([&] {
		if(numStreams != 0) {
			mlopen::deref(handle) = new mlopenContext(numStreams, streams);
		}
		else {
			mlopen::deref(handle) = new mlopenContext();
		}
	});
}

// TODO: Stream size should be a spearate parameter
extern "C"
mlopenStatus_t mlopenGetStream(mlopenHandle_t handle,
		mlopenAcceleratorQueue_t *streamID,
		int numStream) {
	return mlopen::try_([&] {
		mlopen::deref(streamID) = handle->GetStream();
	});
}

extern "C"
mlopenStatus_t mlopenDestroy(mlopenHandle_t handle) {
	return mlopen::try_([&] {
		delete handle;
	});
}
