
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <iostream>

#include "CNVjpegDecoder.h"
#include "helper_cuda.h"
#include "opencv2/opencv.hpp"

typedef std::vector<std::string> FileNames;

using namespace std;

#ifndef PRINT_LOG
#define PRINT_LOG fprintf(stdout, "%s:%d\n", __FILE__, __LINE__); fflush(stdout);
#endif
int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

int dev_free(void *p) { return (int)cudaFree(p); }

int host_malloc(void** p, size_t s, unsigned int f) { return (int)cudaHostAlloc(p, s, f); }
int host_free(void* p) { return (int)cudaFreeHost(p); }




struct decode_params_t {
	std::string input_dir;
	int batch_size;
	int total_images;
	int dev;
	int warmup;

	nvjpegJpegState_t nvjpeg_state;
	nvjpegHandle_t nvjpeg_handle;
	cudaStream_t stream;

	// used with decoupled API
	nvjpegJpegState_t nvjpeg_decoupled_state;
	nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
	nvjpegBufferDevice_t device_buffer;
	nvjpegJpegStream_t  jpeg_streams[2]; //  2 streams for pipelining
	nvjpegDecodeParams_t nvjpeg_decode_params;
	nvjpegJpegDecoder_t nvjpeg_decoder;

	nvjpegOutputFormat_t fmt;
	bool write_decoded;
	std::string output_dir;

	bool pipelined;
	bool batched;
	decode_params_t()
	{
		batched = true;
		pipelined = false;
		dev = 0;
		batch_size = 1;
		warmup = 0;
		//fmt = NVJPEG_OUTPUT_BGR;
	}
};

void destory_nvjpegImage_t(nvjpegImage_t &nvjpg_image)
{
	for (size_t c = 0; c < NVJPEG_MAX_COMPONENT; c++)
	{
		if (nvjpg_image.channel[c]) {
			checkCudaErrors(cudaFree(nvjpg_image.channel[c]));
		}
	}
}





CNVjpegDecoder::CNVjpegDecoder()
{
	_params = (void*)(new decode_params_t);
}
CNVjpegDecoder::~CNVjpegDecoder()
{
	decode_params_t* params = (decode_params_t*)_params;
	checkCudaErrors(nvjpegJpegStateDestroy(params->nvjpeg_state));
	checkCudaErrors(nvjpegDestroy(params->nvjpeg_handle));
}
int CNVjpegDecoder::initialize(int gpu_id, int batch_size)
{
	checkCudaErrors(cudaSetDevice(gpu_id));

	decode_params_t* params = (decode_params_t*)_params;
	params->batch_size = batch_size;
	params->dev = gpu_id;
	params->fmt = NVJPEG_OUTPUT_BGR;
	// params->batched = true;

	checkCudaErrors(cudaSetDevice(gpu_id));

	checkCudaErrors(cudaStreamCreateWithFlags(&params->stream, cudaStreamNonBlocking));

	nvjpegDevAllocator_t dev_allocator = { &dev_malloc, &dev_free };
	nvjpegPinnedAllocator_t pinned_allocator = { &host_malloc, &host_free };
	int flags = 0;
	checkCudaErrors(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator,
		&pinned_allocator, flags, &params->nvjpeg_handle));

	checkCudaErrors(
		nvjpegJpegStateCreate(params->nvjpeg_handle, &params->nvjpeg_state));
	checkCudaErrors(
		nvjpegDecodeBatchedInitialize(params->nvjpeg_handle, params->nvjpeg_state,
			params->batch_size, 1, params->fmt));



	return 0;
}

int decode_images(const FileData &img_data, const std::vector<size_t> &img_len,
	std::vector<nvjpegImage_t> &out, decode_params_t &params,
	double &time) {
	checkCudaErrors(cudaStreamSynchronize(params.stream));
	cudaEvent_t startEvent = NULL, stopEvent = NULL;
	float loopTime = 0;

	checkCudaErrors(cudaEventCreate(&startEvent));
	checkCudaErrors(cudaEventCreate(&stopEvent));

	if (!params.batched) {
		if (!params.pipelined)  // decode one image at a time
		{
			checkCudaErrors(cudaEventRecord(startEvent, params.stream));
			for (int i = 0; i < img_data.size(); i++) {
				checkCudaErrors(nvjpegDecode(params.nvjpeg_handle, params.nvjpeg_state,
					(const unsigned char *)img_data[i].data(),
					img_len[i], params.fmt, &out[i],
					params.stream));
			}
			checkCudaErrors(cudaEventRecord(stopEvent, params.stream));
		}
		else {
			// use de-coupled API in pipelined mode
			checkCudaErrors(cudaEventRecord(startEvent, params.stream));
			checkCudaErrors(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
			int buffer_index = 0;
			checkCudaErrors(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params, params.fmt));
			for (int i = 0; i < params.batch_size; i++) {
				checkCudaErrors(
					nvjpegJpegStreamParse(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i],
						0, 0, params.jpeg_streams[buffer_index]));

				checkCudaErrors(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
					params.pinned_buffers[buffer_index]));

				checkCudaErrors(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
					params.nvjpeg_decode_params, params.jpeg_streams[buffer_index]));

				checkCudaErrors(cudaStreamSynchronize(params.stream));

				checkCudaErrors(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
					params.jpeg_streams[buffer_index], params.stream));

				buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

				checkCudaErrors(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
					&out[i], params.stream));

			}
			checkCudaErrors(cudaEventRecord(stopEvent, params.stream));
		}
	}
	else {
		std::vector<const unsigned char *> raw_inputs;
		for (int i = 0; i < img_data.size(); i++) {
			raw_inputs.push_back((const unsigned char *)img_data[i].data());
		}

		checkCudaErrors(cudaEventRecord(startEvent, params.stream));
		checkCudaErrors(nvjpegDecodeBatched(
			params.nvjpeg_handle, params.nvjpeg_state, raw_inputs.data(),
			img_len.data(), out.data(), params.stream));
		checkCudaErrors(cudaEventRecord(stopEvent, params.stream));

	}
	checkCudaErrors(cudaEventSynchronize(stopEvent));
	checkCudaErrors(cudaEventElapsedTime(&loopTime, startEvent, stopEvent));
	time = static_cast<double>(loopTime);

	return EXIT_SUCCESS;
}

// prepare buffers for RGBi output format
int prepare_buffers(FileData &file_data, 
					std::vector<size_t> &file_len,
					std::vector<int> &img_width,
					std::vector<int> &img_height,
					std::vector<nvjpegImage_t> &ibuf,
					std::vector<nvjpegImage_t> &isz,
					FileNames &current_names,
					decode_params_t &params) 
{
	int widths[NVJPEG_MAX_COMPONENT];
	int heights[NVJPEG_MAX_COMPONENT];
	int channels;
	nvjpegChromaSubsampling_t subsampling;

	for (int i = 0; i < file_data.size(); i++) {
		checkCudaErrors(nvjpegGetImageInfo(
			params.nvjpeg_handle, (unsigned char *)file_data[i].data(), file_len[i],
			&channels, &subsampling, widths, heights));

		img_width[i] = widths[0];
		img_height[i] = heights[0];

		//std::cout << "Processing: " << current_names[i] << std::endl;
		//std::cout << "Image is " << channels << " channels." << std::endl;
		//for (int c = 0; c < channels; c++) {
		//	std::cout << "Channel #" << c << " size: " << widths[c] << " x "
		//		<< heights[c] << std::endl;
		//}
		//switch (subsampling) {
		//case NVJPEG_CSS_444:
		//	std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
		//	break;
		//case NVJPEG_CSS_440:
		//	std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
		//	break;
		//case NVJPEG_CSS_422:
		//	std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
		//	break;
		//case NVJPEG_CSS_420:
		//	std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
		//	break;
		//case NVJPEG_CSS_411:
		//	std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
		//	break;
		//case NVJPEG_CSS_410:
		//	std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
		//	break;
		//case NVJPEG_CSS_GRAY:
		//	std::cout << "Grayscale JPEG " << std::endl;
		//	break;
		//case NVJPEG_CSS_UNKNOWN:
		//	std::cout << "Unknown chroma subsampling" << std::endl;
		//	return EXIT_FAILURE;
		//}

		int mul = 1;
		// in the case of interleaved RGB output, write only to single channel, but
		// 3 samples at once
		if (params.fmt == NVJPEG_OUTPUT_RGBI || params.fmt == NVJPEG_OUTPUT_BGRI) {
			channels = 1;
			mul = 3;
		}
		// in the case of rgb create 3 buffers with sizes of original image
		else if (params.fmt == NVJPEG_OUTPUT_RGB ||
			params.fmt == NVJPEG_OUTPUT_BGR) {
			channels = 3;
			widths[1] = widths[2] = widths[0];
			heights[1] = heights[2] = heights[0];
		}

		// realloc output buffer if required
		for (int c = 0; c < channels; c++) {
			int aw = mul * widths[c];
			int ah = heights[c];
			int sz = aw * ah;
			ibuf[i].pitch[c] = aw;
			if (sz > isz[i].pitch[c]) {
				if (ibuf[i].channel[c]) {
					checkCudaErrors(cudaFree(ibuf[i].channel[c]));
				}
				checkCudaErrors(cudaMalloc((void**)&ibuf[i].channel[c], sz));
				isz[i].pitch[c] = sz;
			}
		}
	}
	return EXIT_SUCCESS;
}

vector<char> CNVjpegDecoder::read_file(const char *file_path)
{
	vector<char> data_buff;
	FILE *fp_img;

	fopen_s(&fp_img, file_path, "rb");
	fseek(fp_img, 0, SEEK_END);
	int nFileSize = ftell(fp_img);
	rewind(fp_img); //reset reading position;

	data_buff.resize(nFileSize);
	int read_size = (int)fread(&data_buff[0], 1, nFileSize, fp_img);
	fclose(fp_img);

	return data_buff;
}

cv::Mat CNVjpegDecoder::imread(const char *filename)
{

	cv::Mat image;
	vector<char> img_data = read_file(filename);
	if (img_data.size() == 0){
		return image;
	}

	



	return image;
}

int CNVjpegDecoder::decode(FileData& filedata, std::vector<cv::Mat> &vImage)
{
	if (filedata.size() == 0)
	{
		return 1;
	}
	decode_params_t* params = (decode_params_t*)_params;

	if (filedata.size() == params->batch_size){
		params->batched = true;
	}else{
		params->batched = false;
	}




	// output buffers
	std::vector<nvjpegImage_t> iout(filedata.size());
	// output buffer sizes, for convenience
	std::vector<nvjpegImage_t> isz(filedata.size());

	std::vector<size_t> file_len(filedata.size());
	FileNames current_names(filedata.size());
	std::vector<int> widths(filedata.size());
	std::vector<int> heights(filedata.size());
	
	for (size_t i = 0; i < filedata.size(); i++) {
		file_len[i] = filedata[i].size();		
		for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
			iout[i].channel[c] = NULL;
			iout[i].pitch[c] = 0;
			isz[i].pitch[c] = 0;

			isz[i].channel[c] = NULL;
			isz[i].pitch[c] = 0;
			isz[i].pitch[c] = 0;
		}
	}
	prepare_buffers(filedata, 
					file_len,
					widths, 
					heights, 
					iout, 
					isz,
					current_names,
					*params);

	double time;
	decode_images(filedata, file_len, iout, *params, time);


	 for (size_t i = 0; i <filedata.size(); i++){
	     int width = widths[i];
	     int height = heights[i];
	     cv::Mat b(cv::Size(widths[i],heights[i]),CV_8UC1);
	     cv::Mat g(cv::Size(widths[i],heights[i]),CV_8UC1);
	     cv::Mat r(cv::Size(widths[i],heights[i]),CV_8UC1);
	     checkCudaErrors(cudaMemcpy2D(b.data, (size_t)width, iout[i].channel[0], (size_t)iout[i].pitch[0],
	                     width, height, cudaMemcpyDeviceToHost));
	     checkCudaErrors(cudaMemcpy2D(g.data, (size_t)width, iout[i].channel[1], (size_t)iout[i].pitch[1],
	                     width, height, cudaMemcpyDeviceToHost));
	     checkCudaErrors(cudaMemcpy2D(r.data, (size_t)width, iout[i].channel[2], (size_t)iout[i].pitch[2],
	                     width, height, cudaMemcpyDeviceToHost));
	     vector<cv::Mat> bgrImage;
	     bgrImage.push_back(b);
	     bgrImage.push_back(g);
	     bgrImage.push_back(r);
	     cv::Mat image;
	     cv::merge(bgrImage,image);
		 vImage.push_back(image);
	 }


	 for (size_t i = 0; i < iout.size(); i++)
	 {
		 destory_nvjpegImage_t(iout[i]);
		 destory_nvjpegImage_t(isz[i]);

	 }





	return 0;
}