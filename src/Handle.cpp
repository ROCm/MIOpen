#include "Handle.hpp"

extern "C" 
mlopenStatus_t mlopenCreate(mlopenHandle_t *handle,
		mlopenStream_t stream = NULL) {
		
	// if handle not valid
	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}

#if MLOpen_BACKEND_OPENCL
	try {
		if(stream != NULL) {
			*handle = new mlopenContext<cl_command_queue>(stream);
		}
		else {
			*handle = new mlopenContext<cl_command_queue>();
			handle->CreateDefaultStream();
		}
	} catch (mlopenStatus_t status) {
		return status;
	}
	
#elif MLOpen_BACKEND_HIP
	try {
		if(stream != NULL) {
			*handle = new mlopenContext<hipStream_t>(stream);
		}
		else {
			*handle = new mlopenContext<hipStream_t>();
			handle->CreateDefaultStream();
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
