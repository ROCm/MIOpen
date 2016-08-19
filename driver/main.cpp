#include <iostream>
#include <cstdio>
#include "driver.hpp"
#include "mloConvHost.hpp"
#include "mloPoolingHost.hpp"
#include "mloNormHost.hpp"
#include "mloNeuronHost.hpp"

void PrintConvParameters(std::vector<int> &in_len,
		std::vector<int> &wei_len,
		std::vector<int> &out_len) {
	printf("\nConvolution Parameters: \n");
	printf("Input Tensor Dimensions:\n");
	for(auto l : in_len)
		printf("%d, ", l);
	
	printf("\nWeight Tensor Dimensions:\n");
	for(auto l : wei_len)
		printf("%d, ", l);
	
	printf("\nOutput Tensor Dimensions:\n");
	for(auto l : out_len)
		printf("%d, ", l);
	printf("\n");
}

int main(int argc, char* argv[]) {
	ConvDriver<float> drv;
	drv.AddCmdLineArgs();
	drv.ParseCmdLineArgs(argc, argv);

	std::vector<int> in_len = drv.GetInputTensorLengthsFromCmdLine();
	std::vector<int> wei_len = drv.GetWeightTensorLengthsFromCmdLine();

	drv.SetInputTensor4d(in_len);
	drv.SetWeightTensor4d(wei_len);
	drv.SetConvDescriptorFromCmdLineArgs();

	std::vector<int> out_len = drv.GetOutputTensorLengths();
	drv.SetOutputTensor4d(out_len);

	if(drv.GetInputFlags().GetValueInt("printconv") == 1) {
		PrintConvParameters(in_len, wei_len, out_len);
	}

	drv.AllocateBuffersAndCopy();
	drv.FindForwardConvAlgo();
	drv.RunForwardConvGPU();

	if(drv.GetInputFlags().GetValueInt("verify") == 1) {
		drv.RunForwardConvCPU();
		drv.VerifyForwardConv();
	}

	// Run backward pass
	if(drv.GetInputFlags().GetValueInt("forwconv") == 0) {
		drv.FindBackwardDataAlgo();
		drv.RunBackwardDataGPU();
	}

	// pooling
	{
		// forward

		mlopenHandle_t handle;
		cl_command_queue q;

		mlopenTensorDescriptor_t inputTensor;
		mlopenTensorDescriptor_t outputTensor;
		mlopenPoolingDescriptor_t poolDesc;

		std::unique_ptr<GPUMem> in_dev;
		std::unique_ptr<GPUMem> out_dev;

		std::vector<float> in;
		std::vector<float> out;
		std::vector<float> outhost;


		handle = drv.GetHandle();
		mlopenGetStream(handle, &q);

		mlopenCreateTensorDescriptor(&inputTensor);
		mlopenCreateTensorDescriptor(&outputTensor);

		mlopenCreatePoolingDescriptor(&poolDesc);

		mlopenPoolingMode_t	mode = mlopenPoolingMax; //mlopenPoolingAverageIncludePadding; // mlopenPoolingMax;
		int	windowHeight = 3;
		int	windowWidth = 3;
		int	pad_h = 1;
		int	pad_w = 1;
		int	u = 2;
		int	v = 2;
		mlopenSet2dPoolingDescriptor(poolDesc, mode, windowHeight, windowWidth, pad_h, pad_w, u, v);

		mlopenSet4dTensorDescriptor(
			inputTensor,
			mlopenFloat,
			UNPACK_VEC4(in_len));

		int out_n, out_c, out_h, out_w;
		mlopenGetPoolingForwardOutputDim(poolDesc, inputTensor, &out_n, &out_c, &out_h, &out_w);
		std::vector<int> out_pooling_len({ out_n, out_c, out_h, out_w });
		mlopenSet4dTensorDescriptor(
			outputTensor,
			mlopenFloat,
			UNPACK_VEC4(out_pooling_len));

		int n_l, c_l, h_l, w_l;
		mlopenGet4dTensorDescriptorLengths(inputTensor, &n_l, &c_l, &h_l, &w_l);

		size_t in_sz = n_l * c_l * h_l *w_l;

		mlopenGet4dTensorDescriptorLengths(outputTensor, &n_l, &c_l, &h_l, &w_l);

		size_t out_sz = n_l * c_l * h_l *w_l;

		cl_context ctx;
		clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

		in_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
		out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

		in = std::vector<float>(in_sz);
		out = std::vector<float>(out_sz, 0);
		outhost = std::vector<float>(out_sz, 0);

		for (int i = 0; i < in_sz; i++) {
			in[i] = rand() * (1.0 / RAND_MAX);
		}


		cl_int status;
		status = in_dev->ToGPU(q, in.data());
		status |= out_dev->ToGPU(q, out.data());

		if (status != CL_SUCCESS)
			printf("Error copying data to GPU\n");
		// forward
		float alpha = 1., beta = 1.;
		mlopenPoolingForward(handle, poolDesc, &alpha, inputTensor, in_dev->GetMem(), &beta, outputTensor, out_dev->GetMem(), false, NULL, 0);

		// verification
		{
			int nInStride, cInStride, hInStride, wInStride;
			mlopenGet4dTensorDescriptorStrides(inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);
			int nIn, cIn, hIn, wIn;
			mlopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
			int nOutStride, cOutStride, hOutStride, wOutStride;
			mlopenGet4dTensorDescriptorStrides(outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);
			int nOut, cOut, hOut, wOut;
			mlopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);

			mlopenPoolingMode_t	mode;
			int	windowHeight;
			int	windowWidth;
			int	pad_h;
			int	pad_w;
			int	u;
			int	v;
			mlopenGet2dPoolingDescriptor(poolDesc, &mode, &windowHeight, &windowWidth, &pad_h, &pad_w, &u, &v);

			int pooling_method = (mode == mlopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;


			status = out_dev->FromGPU(q, out.data());

			status = mloPoolingForwardRunHostAndVerify<float>(
				pooling_method,
				pad_h,
				u,
				windowHeight,
				pad_w,
				v,
				windowWidth,
				nIn,
				cOut,
				hIn,
				wIn,
				hInStride,
				cInStride,
				nInStride,
				hOut,
				wOut,
				hOutStride,
				cOutStride,
				nOutStride,
				in.data(),
				out.data(),
				(1 << 2)
				);

		}

		// backward
		{
			mlopenTensorDescriptor_t dInputTensor;
			mlopenTensorDescriptor_t dOutputTensor;

			std::unique_ptr<GPUMem> din_dev;
			std::unique_ptr<GPUMem> dout_dev;

			std::vector<float> din;
			std::vector<float> dout;
			std::vector<float> dinhost;

			mlopenCreateTensorDescriptor(&dInputTensor);
			mlopenCreateTensorDescriptor(&dOutputTensor);
			mlopenSet4dTensorDescriptor(
				dInputTensor,
				mlopenFloat,
				UNPACK_VEC4(in_len));

			mlopenSet4dTensorDescriptor(
				dOutputTensor,
				mlopenFloat,
				UNPACK_VEC4(out_pooling_len));

			din_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
			dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

			din = std::vector<float>(in_sz);
			dout = std::vector<float>(out_sz, 0);
			dinhost = std::vector<float>(in_sz, 0);


			for (int i = 0; i < out_sz; i++) {
				dout[i] = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
			}

			status = din_dev->ToGPU(q, din.data());
			status |= dout_dev->ToGPU(q, dout.data());
			if (status != CL_SUCCESS)
				printf("Error copying data to GPU\n");


			status = mlopenPoolingBackward(handle,
				poolDesc,
				&alpha,
				outputTensor,
				out_dev->GetMem(),
				dOutputTensor,
				dout_dev->GetMem(),
				inputTensor,
				in_dev->GetMem(),
				&beta,
				dInputTensor,
				din_dev->GetMem(),
				NULL);

			// verification
			{
				int nInStride, cInStride, hInStride, wInStride;
				mlopenGet4dTensorDescriptorStrides(inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);
				int nIn, cIn, hIn, wIn;
				mlopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
				int nOutStride, cOutStride, hOutStride, wOutStride;
				mlopenGet4dTensorDescriptorStrides(outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);
				int nOut, cOut, hOut, wOut;
				mlopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);

				int ndInStride, cdInStride, hdInStride, wdInStride;
				mlopenGet4dTensorDescriptorStrides(dInputTensor, &ndInStride, &cdInStride, &hdInStride, &wdInStride);
				int ndIn, cdIn, hdIn, wdIn;
				mlopenGet4dTensorDescriptorLengths(dInputTensor, &ndIn, &cdIn, &hdIn, &wdIn);
				int ndOutStride, cdOutStride, hdOutStride, wdOutStride;
				mlopenGet4dTensorDescriptorStrides(dOutputTensor, &ndOutStride, &cdOutStride, &hdOutStride, &wdOutStride);
				int ndOut, cdOut, hdOut, wdOut;
				mlopenGet4dTensorDescriptorLengths(dOutputTensor, &ndOut, &cdOut, &hdOut, &wdOut);


				mlopenPoolingMode_t	mode;
				int	windowHeight;
				int	windowWidth;
				int	pad_h;
				int	pad_w;
				int	u;
				int	v;
				mlopenGet2dPoolingDescriptor(poolDesc, &mode, &windowHeight, &windowWidth, &pad_h, &pad_w, &u, &v);

				int pooling_method = (mode == mlopenPoolingMax) ? MLO_POOLING_OP_MAX : MLO_POOLING_OP_AVE;



				status = mloPoolingBackwardRunHost<float>(
					pooling_method,
					pad_h,
					u,
					windowHeight,
					pad_w,
					v,
					windowWidth,
					// host output
					dinhost.data(),
					dout.data(),
					in.data(),
					out.data(),
					ndInStride,
					cdInStride,
					hdInStride,
					nInStride,
					cInStride,
					hInStride,
					wIn,
					hIn,
					cOut,
					nOut,
					ndOutStride,
					cdOutStride,
					hdOutStride,
					wOut,
					hOut,
					nOutStride,
					cOutStride,
					hOutStride
					);

				status = din_dev->FromGPU(q, din.data());

			}
		}
	}

	// LNR
	{
		mlopenHandle_t handle;
		cl_command_queue q;

		mlopenTensorDescriptor_t inputTensor;
		mlopenTensorDescriptor_t outputTensor;
		mlopenLRNDescriptor_t lrnDesc;

		std::unique_ptr<GPUMem> in_dev;
		std::unique_ptr<GPUMem> out_dev;
		std::unique_ptr<GPUMem> scale_dev;

		std::vector<float> in;
		std::vector<float> out;
		std::vector<float> scale;
		std::vector<float> outhost;
		std::vector<float> scalehost;


		handle = drv.GetHandle();
		mlopenGetStream(handle, &q);

		mlopenCreateTensorDescriptor(&inputTensor);
		mlopenCreateTensorDescriptor(&outputTensor);

		mlopenCreateLRNDescriptor(&lrnDesc);


		mlopenLRNMode_t	mode = mlopenLRNWithinChannel; // mlopenLRNCrossChannel;
		unsigned int lrnN = 5;
		double	lrnAlpha = 0.001;
		double	lrnBeta = 0.75;
		double	lrnK = 1.;


		mlopenSetLRNDescriptor(lrnDesc,
			mode,
			lrnN,
			lrnAlpha,
			lrnBeta,
			lrnK);

		mlopenSet4dTensorDescriptor(
			inputTensor,
			mlopenFloat,
			UNPACK_VEC4(in_len));
		mlopenSet4dTensorDescriptor(
			outputTensor,
			mlopenFloat,
			UNPACK_VEC4(in_len));

		bool   do_backward = true;
		size_t	workSpaceSize = 0;

		// get worspace size
		mlopenLRNForward(handle,lrnDesc,NULL,inputTensor,NULL,NULL, outputTensor,NULL, do_backward, NULL, &workSpaceSize);

		int n_l, c_l, h_l, w_l;
		mlopenGet4dTensorDescriptorLengths(inputTensor, &n_l, &c_l, &h_l, &w_l);

		size_t in_sz = n_l * c_l * h_l *w_l;

		mlopenGet4dTensorDescriptorLengths(outputTensor, &n_l, &c_l, &h_l, &w_l);

		size_t out_sz = n_l * c_l * h_l *w_l;

		cl_context ctx;
		clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, NULL);

		in_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
		out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));
		if (do_backward)
		{
			scale_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, workSpaceSize/sizeof(float), sizeof(float)));
		}

		in = std::vector<float>(in_sz);
		out = std::vector<float>(out_sz, 0);
		scale = std::vector<float>(workSpaceSize/sizeof(float), 0);
		outhost = std::vector<float>(out_sz, 0);
		scalehost = std::vector<float>(workSpaceSize / sizeof(float), 0);

		for (int i = 0; i < in_sz; i++) {
			in[i] = rand() * (1.0 / RAND_MAX);
		}


		cl_int status;
		status = in_dev->ToGPU(q, in.data());
		status |= scale_dev->ToGPU(q, scale.data());
		status |= out_dev->ToGPU(q, out.data());

		if (status != CL_SUCCESS)
			printf("Error copying data to GPU\n");
		// forward
		float alpha = 1., beta = 1.;
		// real run
		mlopenLRNForward(handle, lrnDesc, &alpha,
			inputTensor, in_dev->GetMem(),
			&beta,
			outputTensor, out_dev->GetMem(),
			do_backward,
			scale_dev->GetMem(),
			NULL);

		// verification
		{
			int nInStride, cInStride, hInStride, wInStride;
			mlopenGet4dTensorDescriptorStrides(inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);
			int nIn, cIn, hIn, wIn;
			mlopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
			int nOutStride, cOutStride, hOutStride, wOutStride;
			mlopenGet4dTensorDescriptorStrides(outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);
			int nOut, cOut, hOut, wOut;
			mlopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);
			mlopenLRNMode_t	v_mode;
			unsigned int v_lrnN;
			double	v_lrnAlpha;
			double	v_lrnBeta;
			double	v_lrnK;

			mlopenGetLRNDescriptor(lrnDesc,
				&v_mode,
				&v_lrnN,
				&v_lrnAlpha,
				&v_lrnBeta,
				&v_lrnK);

			float alphaoverarea = (v_mode == mlopenLRNCrossChannel) ? v_lrnAlpha / v_lrnN : v_lrnAlpha / (v_lrnN*v_lrnN);

			int pre_pad = (v_lrnN - 1) / 2;
			int pad = v_lrnN - pre_pad - 1;

			int batch_sz = nIn;
			int n_inputs = cIn;
			int bot_height = hIn;
			int bot_width = wIn;
			int bot_stride = hInStride;
			int bot_channel_stride = cInStride;
			int bot_batch_stride = nInStride;


			int n_outputs = cOut;
			int top_height = hOut;
			int top_width = wOut;
			int top_stride = hOutStride;
			int top_channel_stride = cOutStride;
			int	top_batch_stride = nOutStride;


			int top_v_stride = hOutStride;
			int top_v_channel_stride = cOutStride;
			int	top_v_batch_stride = nOutStride;
			int scale_v_stride = top_v_stride;
			int scale_v_channel_stride = top_v_channel_stride;
			int scale_v_batch_stride = top_v_batch_stride;


			mloLRNForwardRunHost<float>(
				do_backward,
				v_mode,
				pad,
				v_lrnN,
				alphaoverarea,
				(float)v_lrnAlpha,
				(float)v_lrnBeta,
				(float)v_lrnK,
				batch_sz,
				n_outputs,
				n_inputs,
				bot_height,
				bot_width,
				bot_stride,
				bot_channel_stride,
				bot_batch_stride,
				top_height,
				top_width,
				top_v_stride,
				top_v_channel_stride,
				top_v_batch_stride,
				scale_v_stride,
				scale_v_channel_stride,
				scale_v_batch_stride,
				in.data(),
				scalehost.data(),
				outhost.data()
				);
	// errors
			bool match = true;
			status = out_dev->FromGPU(q, out.data());
			if (do_backward)
			{
				status |= scale_dev->FromGPU(q, scale.data());
			}

			{
				const double allowedEps = (1 << 2);
				double max_sqr = 1. / 100000000;
				double max_abs_diff = 1. / 100000000;
				bool get_error_pos = true;


				match = mloVerify<float>(
					batch_sz,
					n_outputs,
					top_height,
					top_width,
					top_v_batch_stride,
					top_v_channel_stride,
					top_v_stride,
					top_batch_stride,
					top_channel_stride,
					top_stride,
					outhost.data(),
					out.data(),
					allowedEps,
					max_abs_diff,
					max_sqr,
					get_error_pos
					);
			}
		}

		// backward
		{
			mlopenTensorDescriptor_t dInputTensor;
			mlopenTensorDescriptor_t dOutputTensor;

			std::unique_ptr<GPUMem> din_dev;
			std::unique_ptr<GPUMem> dout_dev;

			std::vector<float> din;
			std::vector<float> dout;
			std::vector<float> dinhost;

			mlopenCreateTensorDescriptor(&dInputTensor);
			mlopenCreateTensorDescriptor(&dOutputTensor);
			mlopenSet4dTensorDescriptor(
				dInputTensor,
				mlopenFloat,
				UNPACK_VEC4(in_len));

			mlopenSet4dTensorDescriptor(
				dOutputTensor,
				mlopenFloat,
				UNPACK_VEC4(in_len));

			din_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(float)));
			dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(float)));

			din = std::vector<float>(in_sz);
			dout = std::vector<float>(out_sz, 0);
			dinhost = std::vector<float>(in_sz, 0);


			for (int i = 0; i < out_sz; i++) {
				dout[i] = (double)(rand() * (1.0 / RAND_MAX) - 0.5) * 0.001;
			}

			status = din_dev->ToGPU(q, din.data());
			status |= dout_dev->ToGPU(q, dout.data());
			if (status != CL_SUCCESS)
				printf("Error copying data to GPU\n");

			// backward
			float alpha = 1., beta = 1.;
			// real run
			mlopenLRNBackward(handle,
				lrnDesc,
				&alpha,
				outputTensor, out_dev->GetMem(),
				dOutputTensor, dout_dev->GetMem(),
				inputTensor, in_dev->GetMem(),
				&beta,
				dInputTensor, din_dev->GetMem(),
				scale_dev->GetMem());
			// verification
			{
				int nInStride, cInStride, hInStride, wInStride;
				mlopenGet4dTensorDescriptorStrides(inputTensor, &nInStride, &cInStride, &hInStride, &wInStride);
				int nIn, cIn, hIn, wIn;
				mlopenGet4dTensorDescriptorLengths(inputTensor, &nIn, &cIn, &hIn, &wIn);
				int nOutStride, cOutStride, hOutStride, wOutStride;
				mlopenGet4dTensorDescriptorStrides(outputTensor, &nOutStride, &cOutStride, &hOutStride, &wOutStride);
				int nOut, cOut, hOut, wOut;
				mlopenGet4dTensorDescriptorLengths(outputTensor, &nOut, &cOut, &hOut, &wOut);


				int ndInStride, cdInStride, hdInStride, wdInStride;
				mlopenGet4dTensorDescriptorStrides(dInputTensor, &ndInStride, &cdInStride, &hdInStride, &wdInStride);
				int ndIn, cdIn, hdIn, wdIn;
				mlopenGet4dTensorDescriptorLengths(dInputTensor, &ndIn, &cdIn, &hdIn, &wdIn);
				int ndOutStride, cdOutStride, hdOutStride, wdOutStride;
				mlopenGet4dTensorDescriptorStrides(dOutputTensor, &ndOutStride, &cdOutStride, &hdOutStride, &wdOutStride);
				int ndOut, cdOut, hdOut, wdOut;
				mlopenGet4dTensorDescriptorLengths(dOutputTensor, &ndOut, &cdOut, &hdOut, &wdOut);


				mlopenLRNMode_t	v_mode;
				unsigned int v_lrnN;
				double	v_lrnAlpha;
				double	v_lrnBeta;
				double	v_lrnK;

				mlopenGetLRNDescriptor(lrnDesc,
					&v_mode,
					&v_lrnN,
					&v_lrnAlpha,
					&v_lrnBeta,
					&v_lrnK);

				float alphaoverarea = (v_mode == mlopenLRNCrossChannel) ? v_lrnAlpha / v_lrnN : v_lrnAlpha / (v_lrnN*v_lrnN);

				int pre_pad = (v_lrnN - 1) / 2;
				int pad = v_lrnN - pre_pad - 1;

				int batch_sz = nIn;
				int n_inputs = cIn;
				int bot_height = hIn;
				int bot_width = wIn;
				int bot_stride = hInStride;
				int bot_channel_stride = cInStride;
				int bot_batch_stride = nInStride;

				int bot_df_stride = hdInStride;
				int bot_df_channel_stride = cdInStride;
				int bot_df_batch_stride = ndInStride;

				int bot_df_v_stride = hdInStride;
				int bot_df_v_channel_stride = cdInStride;
				int bot_df_v_batch_stride = ndInStride;


				int n_outputs = cOut;
				int top_height = hOut;
				int top_width = wOut;
				int top_stride = hOutStride;
				int top_channel_stride = cOutStride;
				int	top_batch_stride = nOutStride;

				int top_df_stride = hdOutStride;
				int top_df_channel_stride = cdOutStride;
				int	top_df_batch_stride = ndOutStride;

				int scale_stride = top_df_stride;
				int scale_channel_stride = top_df_channel_stride;
				int scale_batch_stride = top_df_batch_stride;


				status = mloLRNBackwardRunHost<float>(
					(int)v_mode,
					pad,
					v_lrnN,
					alphaoverarea,
					(float)v_lrnAlpha,
					(float)v_lrnBeta,
					(float)v_lrnK,
					batch_sz,
					n_outputs,
					n_inputs,
					bot_height,
					bot_width,
					bot_stride,
					bot_channel_stride,
					bot_batch_stride,
					bot_df_v_stride,
					bot_df_v_channel_stride,
					bot_df_v_batch_stride,
					top_height,
					top_width,
					top_stride,
					top_channel_stride,
					top_batch_stride,
					top_df_stride,
					top_df_channel_stride,
					top_df_batch_stride,
					scale_stride,
					scale_channel_stride,
					scale_batch_stride,
					out.data(),
					dout.data(),
					scale.data(),
					in.data(),
					dinhost.data()
					);

				// errors
				bool match = true;
				status = din_dev->FromGPU(q, din.data());
				{

					double sqr_accum = 0;
					double max_err = -std::numeric_limits<double>::min();
					double allowedEps = 4;
					double max_abs_diff = 0.00000001;
					double max_sqr = 0.000000001;
					bool get_error_pos = true;

					match = mloVerify<float>(
						batch_sz,
						n_inputs,
						bot_height,
						bot_width,
						bot_df_v_batch_stride,
						bot_df_v_channel_stride,
						bot_df_v_stride,
						bot_df_batch_stride,
						bot_df_channel_stride,
						bot_df_stride,
						dinhost.data(),
						din.data(),
						allowedEps,
						max_abs_diff,
						max_sqr,
						get_error_pos
						);

				}
			}
		} // LRN backward
	}

	return 0;
}
