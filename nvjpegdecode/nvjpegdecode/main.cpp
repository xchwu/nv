#include <iostream>
#include <stdio.h>
#include "CNVjpegDecoder.h"
using namespace std;

#ifdef _WIN32
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "nvjpeg.lib")
#ifdef _DEBUG
#pragma comment(lib, "opencv_world310d.lib")
#else
#pragma comment(lib, "opencv_world310.lib")
#endif
#endif


vector<char>  read_file(char *file_path)
{
	vector<char> data_buff;
	FILE *fp_img;

	fopen_s(&fp_img,file_path, "rb");
	fseek(fp_img, 0, SEEK_END);
	int nFileSize = ftell(fp_img);
	rewind(fp_img); //reset reading position;

	data_buff.resize(nFileSize);
	int read_size = (int)fread(&data_buff[0], 1, nFileSize, fp_img);	
	fclose(fp_img);

	return data_buff;
}

int main(int argc, char* argv[])
{
	CNVjpegDecoder nvjpeg_decoder;
	nvjpeg_decoder.initialize(0, 16);
	FileData filedata;

	for (size_t i = 0; i < 16; i++)
	{
		char filename[100];
		sprintf_s(filename,100, "%02d.jpg", i);
		vector<char> img_data = read_file(filename);
		filedata.push_back(img_data);
	}

	cv::Mat img = nvjpeg_decoder.imread("00.jpg");

	vector<cv::Mat> vImage;
	nvjpeg_decoder.decode(filedata, vImage);

	for (cv::Mat image : vImage)
	{
		cv::imshow("image", image);
		cout << image.size() << endl;
		cv::waitKey();
	}
	
	return 0;
}