#ifndef GUARD_MIOPEN_CONTEXT_HPP_
#define GUARD_MIOPEN_CONTEXT_HPP_

#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <miopen/common.hpp>
#include <miopen/kernel.hpp>
#include <vector>
#include <cstdio>
#include <cstring>
#include <memory>

namespace miopen {

struct HandleImpl;

struct Handle : miopenHandle {
	
	Handle();
	Handle(miopenAcceleratorQueue_t stream);
    Handle(Handle&&) noexcept;
	~Handle();

	miopenAcceleratorQueue_t GetStream() const;
	void SetStream(miopenAcceleratorQueue_t streamID) const;

    void EnableProfiling(bool enable=true);
	
	void ResetKernelTime();
	void AccumKernelTime(float x);

    float GetKernelTime() const;
	bool IsProfilingEnabled() const;

    KernelInvoke GetKernel(
            const std::string& algorithm,
            const std::string& network_config,
            const std::string& program_name,
            const std::string& kernel_name,
            const std::vector<size_t>& vld,
            const std::vector<size_t>& vgd,
            const std::string& params);

    KernelInvoke GetKernel(
        const std::string& algorithm,
        const std::string& network_config);

    Program LoadProgram(const std::string &program_name, std::string params, bool is_kernel_str);

    void Finish() const;
    void Flush() const;

    std::size_t GetLocalMemorySize();
    std::size_t GetMaxComputeUnits();

    std::string GetDeviceName();

    void Copy(ConstData_t src, Data_t dest, std::size_t size);

	ManageDataPtr Create(std::size_t sz);
	ManageDataPtr& WriteTo(const void* data, ManageDataPtr& ddata, std::size_t sz);
    void ReadTo(void* data, const ManageDataPtr& ddata, std::size_t sz);

	template<class T>
	ManageDataPtr Create(std::size_t sz)
	{
		return this->Create(sz*sizeof(T));
	}

	template<class Container>
    ManageDataPtr Write(const Container& c)
    {
    	using type = typename Container::value_type;
    	auto buf = this->Create<type>(c.size());
    	return std::move(this->WriteTo(reinterpret_cast<const void*>(c.data()), buf, c.size()*sizeof(type)));
    }

    template<class T>
    std::vector<T> Read(const ManageDataPtr& ddata, std::size_t sz)
    {
    	std::vector<T> result(sz);
    	this->ReadTo(result.data(), ddata, sz*sizeof(T));
    	return result;
    }

	std::unique_ptr<HandleImpl> impl;
	
};
} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenHandle, miopen::Handle);


#endif // GUARD_MIOPEN_CONTEXT_HPP_
