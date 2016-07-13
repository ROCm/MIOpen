#pragma once
#ifndef _OCL_HELPER_HPP_
#define _OCL_HELPER_HPP_

#include "MLOpen.h"
#include <string>
#include <iostream>

class CLHelper {

	public:

	static mlopenStatus_t LoadProgramFromSource(cl_program &program,
			cl_command_queue &queue,
			const std::string &program_name);

	static mlopenStatus_t BuildProgram(cl_program &program,
			cl_command_queue &queue,
			const std::string &params);

	static mlopenStatus_t CreateKernel(cl_program &program,
			cl_kernel &kernel,
			const std::string &kernel_name);

	static void CheckCLStatus(cl_int status, const std::string &desc);
};
#endif // _OCL_HELPER_HPP_
