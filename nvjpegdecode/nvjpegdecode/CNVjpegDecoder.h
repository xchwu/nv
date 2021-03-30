#ifndef __CNVJPEG_H__
#define __CNVJPEG_H__

#include <vector>
#include <opencv2/opencv.hpp>

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

	cv::Mat imread(const char *filename);
protected:
	std::vector<char> read_file(const char *filename);
private:

	void *_params;
};




#endif