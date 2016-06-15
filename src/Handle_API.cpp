#include "Handle.hpp"
#include <cstdio>

extern "C" 
mlopenStatus_t mlopenCreate(mlopenHandle_t *handle,
		mlopenStream_t stream ) {
		
	printf("In mlopenCreate\n");
	// if handle not valid
	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}

#if MLOpen_BACKEND_OPENCL
	try {
		if(stream != NULL) {
			*handle = new mlopenContext(stream);
		}
		else {
			*handle = new mlopenContext();
			(*handle)->CreateDefaultStream<cl_command_queue>();
		}
	} catch (mlopenStatus_t status) {
		return status;
	}
	
#elif MLOpen_BACKEND_HIP
	try {
		if(stream != NULL) {
			*handle = new mlopenContext(stream);
		}
		else {
			*handle = new mlopenContext();
			(*handle)->CreateDefaultStream<hipStream_t>();
		}
	} catch (mlopenStatus_t status) {
		return status;
	}
#endif
	
	return mlopenStatusSuccess;
}

extern "C" 
mlopenStatus_t mlopenSetStream(mlopenHandle_t handle, 
		mlopenStream_t streamID) {
	printf("In mlopenSetStream\n");
	return handle->SetStream(streamID);
}

extern "C"
mlopenStatus_t mlopenGetStream(mlopenHandle_t handle,
		mlopenStream_t *streamID) {
	return handle->GetStream(streamID);
}

extern "C"
mlopenStatus_t mlopenDestroy(mlopenHandle_t handle) {
	try {
		delete handle;
	} catch (mlopenStatus_t status) {
		return status;
	}
	return mlopenStatusSuccess;
}
