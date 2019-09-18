//
// Created by marco on 22/08/19.
//

#include "cudaSobel.h"
#include <iostream>
#include <algorithm>

#define GRIDVAL 32

__global__ void _sobel_process_kernel_(unsigned char* d_src, unsigned char* d_dst, int row, int col)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float dx, dy;
    if( x > 0 && y > 0 && x < col-1 && y < row-1) {
        dx = (-1* d_src[(y-1)*col + (x-1)]) + (-2*d_src[y*col+(x-1)]) + (-1*d_src[(y+1)*col+(x-1)]) +
             (    d_src[(y-1)*col + (x+1)]) + ( 2*d_src[y*col+(x+1)]) + (   d_src[(y+1)*col+(x+1)]);
        dy = (    d_src[(y-1)*col + (x-1)]) + ( 2*d_src[(y-1)*col+x]) + (   d_src[(y-1)*col+(x+1)]) +
             (-1* d_src[(y+1)*col + (x-1)]) + (-2*d_src[(y+1)*col+x]) + (-1*d_src[(y+1)*col+(x+1)]);
    }
    d_dst[y*col + x] = static_cast<unsigned char>(sqrt( (dx*dx) + (dy*dy) ));
}

cudaSobel::cudaSobel(cv::Mat inImage, const char *imgName) {
    inputImage = inImage;
    inputImageFileName = imgName;
    outputImage = cv::Mat(inputImage.size(),inputImage.type(),cvScalar(0));
}

/***
 * starts parallel Sobel computation with CUDA
 *
 */

void cudaSobel::computeCuda() {

    cv::Mat I = inputImage; // shorter for convolution :)
    int w = I.cols; // number of columns
    int h = I.rows; // number of rows

    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;

    const size_t ARRAY_BYTES = h * w * sizeof(unsigned char);

    cudaMalloc((void**) &d_src, ARRAY_BYTES);
    cudaMalloc((void**) &d_dst, ARRAY_BYTES);

    cudaMemcpy(d_src, inputImage.data, ARRAY_BYTES, cudaMemcpyHostToDevice);

    dim3 threads(GRIDVAL, GRIDVAL);
    dim3 blocks(w / threads.x + 1, h / threads.y + 1);

    _sobel_process_kernel_<<<blocks, threads>>>(d_src, d_dst, h, w);

    // waits until is done
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_dst, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
}

/***
 *
 * @return inputImage file name
 */
const char* cudaSobel::getInputImageFileName(){
    return inputImageFileName;
}

/***
 * get computation time for the given image
 * @return computation time in microseconds
 */
double cudaSobel::getComputationTime() {
    return executionTime;
}

/***
 * display the result of edges detection
 * @param title window title
 */
void cudaSobel::displayOutputImg(const cv::String title) {
    cv::namedWindow(title, CV_WINDOW_NORMAL);
    imshow(title, outputImage);
    cv::waitKey(0); // waits for user's input
}