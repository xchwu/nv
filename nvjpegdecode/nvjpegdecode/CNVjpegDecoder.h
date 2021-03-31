#ifndef __CNVJPEG_H__
#define __CNVJPEG_H__

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>


typedef std::vector<std::string> FileNames;
typedef std::vector<std::vector<char> > FileData;


class CNVjpegDecoder
{
public:
	CNVjpegDecoder();
	~CNVjpegDecoder();

	int initialize(int gpu_id, int batch_size);
	/*	
	error code:
		0 success
		1 filedata empty
	*/
	int decode(FileData& filedata, std::vector<cv::Mat> &vImage); 

	int decode(FileNames &image_names, std::vector<cv::Mat> &vImage);

	cv::Mat imread(const char *filename);
protected:
	std::vector<char> read_file(const char *filename);
	// output buffers
	std::vector<nvjpegImage_t> iout;
	// output buffer sizes, for convenience
	std::vector<nvjpegImage_t> isz;
private:

	void *_params;
};




#endif