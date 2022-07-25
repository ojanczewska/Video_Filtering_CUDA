
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <iostream>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;

__global__ void bgr_to_gray_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep);

__global__ void sobel_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep);

__global__ void convolution_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep, float* macierz);

__global__ void negative_kernel(unsigned char* input,
	unsigned char* output,
	int szer,
	int wys,
	int colorWidthStep);

void convolution(const Mat& input, Mat& output, float* macierz);
void sobel(const Mat& input, Mat& output);
void negative(const Mat& input, Mat& output);


void convert_to_gray(const Mat& input, Mat& output);

int main(int argc, char** argv)
{

	Mat image, image2, combine;

	VideoCapture cap(0);

	// -------- ZMIANA ROZMIARU ------------

	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 680);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);


	if (!cap.isOpened())
	{ // check if video device has been initialised
		std::cout << "cannot open camera";
	}


	float GaussianBlur[9] = { 1, 2, 1,2,4,2,1,2,1 };

	for (int i = 0; i < 9; i++)
	{
		float x = GaussianBlur[i];
		GaussianBlur[i] = float((x / 16.0));
	}


	float BoxBlur[9] = { 1,1,1,1,1,1,1,1,1 };
	for (int i = 0; i < 9; i++)
	{
		float x = BoxBlur[i];
		BoxBlur[i] = float((x / 9));
	}
	float  Sharpen[9] = { 0, -1, 0,-1,5,-1,0,-1,0 };
	float EdgeDetection[9] = { -1, -1, -1,-1,8,-1, -1, -1,-1 };


	int decision;
	cout << "Select the type of fiter you want to apply : " << endl << "1. Edge detection with a Sobel filter" << endl <<
		"2. Monochrome filter" << endl << "3. Edge detection with convolution filter" << endl <<
		"4. GaussianBlur filter" << endl << "5. BoxBlur filter" << endl << "6. Image sharpening" << endl << "7. Negative" << endl;
	cin >> decision;
	switch (decision)
	{
	case 1:
	case 2:
	case 3:
	case 4:
	case 5:
	case 6:
	case 7:
		break;
	default:
		cout << "Wrong number!" << endl;
		return 0;
		break;
	}



	while (true) {

		cap >> image;
		cap >> image2;

		resize(image, image, Size(460, 460));
		resize(image2, image2, Size(460, 460));


		switch (decision)
		{
		case 1:
			sobel(image, image);
			break;
		case 2:

			convert_to_gray(image, image);
			break;
		case 3:
			convolution(image, image, EdgeDetection);
			break;
		case 4:
			convolution(image, image, GaussianBlur);
			convolution(image, image, GaussianBlur);
			convolution(image, image, GaussianBlur);
			break;
		case 5:
			convolution(image, image, BoxBlur);
			convolution(image, image, BoxBlur);
			convolution(image, image, BoxBlur);
			break;
		case 6:
			convolution(image, image, Sharpen);
			break;
		case 7:
			negative(image, image);
			break;

		}

		hconcat(image, image2, combine);

		imshow("Display window", combine);
		waitKey(25);
	}


	//-----------  FPS  -----------------

	//for (int licz = 60; licz < 4160; licz +=100) {

	//	auto start = chrono::steady_clock::now();

	//	cout << "ROZMIAR = " << licz << endl;
	//	for (int i = 0; i < 10; i++) {

	//		cap >> image;

	//		resize(image, image, Size(licz, licz));

	//		//for (int x = 0; x < licz; x++) {
	//			konwolucja(image, image, Sharpen);
	//		//}
	//		
	//	}
	//	auto end = chrono::steady_clock::now();

	//	chrono::duration<double> elapsed = end - start;

	//	
	//	double fps2 = 10 / elapsed.count();
	//	cout << "estinmated fps = " << fps2 << endl;
	//}


	return 0;
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	const int sizeInput = input.step * input.rows;
	const int sizeOutput = output.step * output.rows;

	unsigned char* d_input, *d_output;

	cudaMalloc<unsigned char>(&d_input, sizeInput);
	cudaMalloc<unsigned char>(&d_output, sizeOutput);

	cudaMemcpy(d_input, input.ptr(), sizeInput, cudaMemcpyHostToDevice);

	const dim3 block(16, 16);
	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	bgr_to_gray_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step);

	cudaMemcpy(output.ptr(), d_output, sizeOutput, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}

void sobel(const Mat& input, Mat& output)
{

	const int sizeInput = input.step * input.rows;
	const int sizeOutput = output.step * output.rows;

	unsigned char* d_input, *d_output;
	cudaMalloc<unsigned char>(&d_input, sizeInput);
	cudaMalloc<unsigned char>(&d_output, sizeOutput);

	cudaMemcpy(d_input, input.ptr(), sizeInput, cudaMemcpyHostToDevice);
	const dim3 block(16, 16);

	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	sobel_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step);

	cudaMemcpy(output.ptr(), d_output, sizeOutput, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

}
void negative(const Mat& input, Mat& output) {
	const int sizeInput = input.step * input.rows;
	const int sizeOutput = output.step * output.rows;

	unsigned char* d_input, *d_output;
	cudaMalloc<unsigned char>(&d_input, sizeInput);
	cudaMalloc<unsigned char>(&d_output, sizeOutput);

	cudaMemcpy(d_input, input.ptr(), sizeInput, cudaMemcpyHostToDevice);
	const dim3 block(16, 16);

	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	negative_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step);

	cudaMemcpy(output.ptr(), d_output, sizeOutput, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

}
void convolution(const Mat& input, Mat& output, float* macierz)
{
	const int sizeInput = input.step * input.rows;
	const int sizeOutput = output.step * output.rows;

	unsigned char* d_input, *d_output;
	float* d_matrix;
	cudaMalloc<unsigned char>(&d_input, sizeInput);
	cudaMalloc<unsigned char>(&d_output, sizeOutput);

	int sizeMatrix = 9 * sizeof(float);
	cudaMalloc((void**)&d_matrix, sizeMatrix);

	cudaMemcpy(d_input, input.ptr(), sizeInput, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix, macierz, sizeMatrix, cudaMemcpyHostToDevice);

	const dim3 block(16, 16);

	const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);

	convolution_kernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step, d_matrix);

	cudaMemcpy(output.ptr(), d_output, sizeOutput, cudaMemcpyDeviceToHost);
	cudaFree(d_input); cudaFree(d_output); cudaFree(d_matrix);
}

__global__ void bgr_to_gray_kernel(unsigned char* input,
	unsigned char* output,
	int width,
	int hight,
	int colorWidthStep)
{

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;


	if ((xIndex < width) && (yIndex < hight))
	{

		const int pixel_id = yIndex * colorWidthStep + (3 * xIndex);

		const unsigned char blue = input[pixel_id];
		const unsigned char green = input[pixel_id + 1];
		const unsigned char red = input[pixel_id + 2];

		const float gray = (red * 0.3f + green * 0.59f + blue * 0.11f);

		output[pixel_id] = gray;
		output[pixel_id + 1] = gray;
		output[pixel_id + 2] = gray;

	}
}


__global__ void sobel_kernel(unsigned char* input,
	unsigned char* output,
	int width,
	int hight,
	int colorWidthStep)
{
	int Gx[3][3] = { {-1,0,1 },{-2,0,2},{-1,0,1} };
	int Gy[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex < width - 1 && (xIndex > 0)) && (yIndex < hight - 1 && (yIndex > 0))) {

		const int pixel_id = yIndex * colorWidthStep + (3 * xIndex); 


		float pix_x = (Gx[0][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex - 1))]) +
			(Gx[0][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex - 1))]) +
			(Gx[0][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex - 1))]) +
			(Gx[1][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex))]) +
			(Gx[1][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex))]) +
			(Gx[1][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex))]) +
			(Gx[2][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex + 1))]) +
			(Gx[2][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex + 1))]) +
			(Gx[2][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex + 1))]);

		float pix_y = (Gy[0][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex - 1))]) +
			(Gy[0][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex - 1))]) +
			(Gy[0][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex - 1))]) +
			(Gy[1][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex))]) +
			(Gy[1][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex))]) +
			(Gy[1][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex))]) +
			(Gy[2][0] * input[(yIndex - 1) * colorWidthStep + (3 * (xIndex + 1))]) +
			(Gy[2][1] * input[(yIndex)*colorWidthStep + (3 * (xIndex + 1))]) +
			(Gy[2][2] * input[(yIndex + 1) * colorWidthStep + (3 * (xIndex + 1))]);

		float val1 = sqrtf((pix_x * pix_x) + (pix_y * pix_y));

		unsigned char val = static_cast<unsigned char>(val1);

		output[pixel_id] = val;
		output[pixel_id + 1] = val;
		output[pixel_id + 2] = val;
	}


}

__global__ void convolution_kernel(unsigned char* input,
	unsigned char* output,
	int width,
	int hight,
	int colorWidthStep, float* matrix)
{

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	__syncthreads();

	if ((xIndex < width - 3 && (xIndex > 2)) && (yIndex < hight - 3 && (yIndex > 2))) {


		const int pixel_id = yIndex * colorWidthStep + (3 * xIndex); 
		float val_r = 0;
		float val_g = 0;
		float val_b = 0;
		int r = 0;
		int c = 0;
		for (int i = 0; i < 3; i++)
		{
			r = -1 + i;
			for (int j = 0; j < 3; j++)
			{
				c = -1 + j;
				val_r += matrix[3 * i + j] * input[(yIndex + c) * colorWidthStep + (3 * (xIndex + r))];
				val_g += matrix[3 * i + j] * input[(yIndex + c) * colorWidthStep + (3 * (xIndex + r)) + 1];
				val_b += matrix[3 * i + j] * input[(yIndex + c) * colorWidthStep + (3 * (xIndex + r)) + 2];
			}
		}
		unsigned char val_rc = static_cast<unsigned char>(val_r);
		unsigned char val_gc = static_cast<unsigned char>(val_g);
		unsigned char val_bc = static_cast<unsigned char>(val_b);

		output[pixel_id] = val_rc;
		output[pixel_id + 1] = val_gc;
		output[pixel_id + 2] = val_bc;
	}

}

__global__ void negative_kernel(unsigned char* input,
	unsigned char* output,
	int width,
	int hight,
	int colorWidthStep)
{
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	__syncthreads();

	if ((xIndex < width && (xIndex > 0)) && (yIndex < hight && (yIndex > 0))) {

		const int pixel_id = yIndex * colorWidthStep + (3 * xIndex);
		float val_r = 0;
		float val_g = 0;
		float val_b = 0;

		val_r = 255 - input[pixel_id];
		val_g = 255 - input[pixel_id + 1];
		val_b = 255 - input[pixel_id + 2];

		unsigned char val_rc = static_cast<unsigned char>(val_r);
		unsigned char val_gc = static_cast<unsigned char>(val_g);
		unsigned char val_bc = static_cast<unsigned char>(val_b);

		output[pixel_id] = val_rc;
		output[pixel_id + 1] = val_gc;
		output[pixel_id + 2] = val_bc;

	}
}
