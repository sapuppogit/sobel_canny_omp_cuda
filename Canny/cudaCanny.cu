//
// Created by marco on 05/09/19.
//

#include "cudaCanny.h"
#include <omp.h>

#include <vector>
#include <iostream>
#include <fstream>

#define HIGH_THRESHOLD 140
#define LOW_THRESHOLD 70

#define GRIDVAL 16

cudaCanny::cudaCanny(cv::Mat inImage, const char *imgName, int size, double sigma) {
    inputImage = inImage;
    inputImageFileName = imgName;
    generateFilter(size, sigma); // create filter
}

__global__ void _canny_apply_filter_(unsigned char* d_src, double* d_filter, unsigned char* d_dst, int filterSize, int filteredSize, int inputSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    if (x >= 0 && y >= 0 && x < filteredSize && y < filteredSize) {
        double sum = 0;

        for (int i = 0; i < filterSize; i++)
            for (int j = 0; j < filterSize; j++) {
                sum += d_filter[i * filterSize + j] * (double) (d_src[(y + i) * inputSize + (x + j)]);
            }
        d_dst[x + y * filteredSize] = sum;
    }
}


__global__ void _canny_angle_map_(unsigned char* d_src, unsigned char* d_dst, unsigned char* d_angleMap, int outputSize, int inputSize)
{

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    double sq;
    if (x >= 0 && y >= 0 && x < outputSize && y < outputSize) {
        float dx, dy;
        dx = (-1 * d_src[y * inputSize + x]) + (-2 * d_src[(y + 1) * inputSize + x]) + (-1 * d_src[(y + 2) * inputSize + x]) +
             (d_src[y * inputSize + (x + 2)]) + (2 * d_src[(y + 1) * inputSize + (x + 2)]) + (d_src[(y + 2) * inputSize + (x + 2)]);
        dy = (d_src[y * inputSize + x]) + (2 * d_src[y * inputSize + (x + 1)]) + (d_src[y * inputSize + (x + 2)]) +
             (-1 * d_src[(y + 2) * inputSize + x]) + (-2 * d_src[(y + 2) * inputSize + (x + 1)]) +
             (-1 * d_src[(y + 2) * inputSize + (x + 2)]);
        sq = sqrt(float((dx * dx) + (dy * dy)));

        if (sq > 255) d_dst[y * outputSize + x] = 255;
        else d_dst[y * outputSize + x] = sq;

        if (dx == 0) d_angleMap[y * outputSize + x] = 90;
        else d_angleMap[y * outputSize + x] = atan(dy / dx);

    }


}


__global__ void _nonmax_suppression_(unsigned char* d_src, unsigned char* d_angleMap, unsigned char* d_dst, int outputSize, int inputSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > 0 && y > 0 && x < outputSize && y < outputSize) {

        float tan = d_angleMap[y*inputSize + x]; // corresponding tangent value in angles map

        d_dst[(y-1)*outputSize + x-1] = d_src[y*inputSize + x];


        //Horizontal Edge
        if (((-22.5 < tan) && (tan <= 22.5)) || ((157.5 < tan) && (tan <= -157.5))) {
            if ((d_src[y*inputSize + x] < d_src[(y+1)*inputSize + x]) ||
                (d_src[y*inputSize + x] < d_src[(y-1)*inputSize + x]))
                d_dst[(y-1)*outputSize + x-1] = 0;
        }
        //Vertical Edge
        if (((-112.5 < tan) && (tan <= -67.5)) || ((67.5 < tan) && (tan <= 112.5))) {
            if ((d_src[y*inputSize + x] < d_src[y*inputSize + x+1]) ||
                (d_src[y*inputSize + x] < d_src[y*inputSize + x-1]))
                d_dst[(y-1)*outputSize + x-1] = 0;
        }

        //-45 Degree Edge
        if (((-67.5 < tan) && (tan <= -22.5)) || ((112.5 < tan) && (tan <= 157.5))) {
            if ((d_src[y*inputSize + x] < d_src[(y+1)*inputSize + x-1]) ||
                (d_src[y*inputSize + x] < d_src[(y-1)*inputSize + x+1]))
                d_dst[(y-1)*outputSize + x-1] = 0;
        }

        //45 Degree Edge
        if (((-157.5 < tan) && (tan <= -112.5)) || ((22.5 < tan) && (tan <= 67.5))) {
            if ((d_src[y*inputSize + x] < d_src[(y+1)*inputSize + x+1]) ||
                (d_src[y*inputSize + x] < d_src[(y-1)*inputSize + x-1]))
                d_dst[(y-1)*outputSize + x-1] = 0;
        }
    }

}

__global__ void _edges_concatenation_(unsigned char* d_src, unsigned char* d_dst, int highthr, int lowthr, int size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > 0 && y > 0 && x < size-1 && y < size-1) {
        int pixelVal = d_src[y * size + x];

        if (pixelVal > highthr) {
            // strong edge
            d_dst[y * size + x] = 255;

        } else if (pixelVal <= highthr && pixelVal >= lowthr) {

            // is connected to a strong edge?
            // check if region is feasible ( 8-bit neighbours)
            // check neighbours
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {

                    // region is feasible
                    int pVal = d_src[(y+j) * size + i+x];
                    if (pVal >= highthr) {
                        d_dst[y * size + x] = 255; // connected to a strong edge
                        break;

                    } else if (pVal < lowthr) {
                        d_dst[y * size + x] = 0;
                        break;
                    }
                }
            }


        } else if (pixelVal < lowthr) {
            d_dst[y * size + x] = 0; // suppression
        }
    }
    else d_dst[y * size + x] = 0;
}


/***
 * creates a gaussian filter of the given size with the specified sigma
 * @param size
 * @param sigma
 * @return returns the gaussian filter
 */
void cudaCanny::generateFilter(int size, double sigma) {

    std::vector<double> filter(size*size); // output filter (size*size)

    double r, s = 2.0 * sigma * sigma;
    double sum = 0; // for filter normalization

    // fill the filter
    for (int y = 0 ; y < size; y++) {
        for (int x = 0; x < size; x++) {
            r = sqrt((y-size/2)*(y-size/2) + (x-size/2)*(x-size/2) );
            filter[y*size + x] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += filter[y*size + x];
        }
    }

    // normalize elements from 0 to 1
    for (int i = 0; i < size*size; i++) {
            filter[i] /= sum;
    }

    gaussianFilter = filter;
}


cv::Mat cudaCanny::computeCuda() {

    // STEP 1: GAUSSIAN FILTER
    int size = (int)sqrt(gaussianFilter.size())/2;
    cv::Mat gaussianFiltered = cv::Mat(inputImage.rows - 2*size, inputImage.cols - 2*size, CV_8UC1, cv::Scalar(0)); // creates an empty output image

    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    const size_t ARRAY_BYTES = inputImage.cols * inputImage.rows * sizeof(unsigned char);

    double* d_filter = &gaussianFilter[0];
    cudaMalloc((void**) &d_src, ARRAY_BYTES);
    cudaMalloc((void**) &d_filter, gaussianFilter.size()*sizeof(double));
    cudaMalloc((void**) &d_dst, (inputImage.cols - 2*size) * (inputImage.rows - 2*size) * sizeof(unsigned char));

    cudaMemcpy(d_src, inputImage.data, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, gaussianFilter.data(), gaussianFilter.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, gaussianFiltered.data, (inputImage.cols - 2*size) * (inputImage.rows - 2*size) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threads(GRIDVAL,GRIDVAL);
    dim3 blocks((inputImage.cols - 2*size) / GRIDVAL + 1, (inputImage.rows - 2*size) / GRIDVAL + 1);

    _canny_apply_filter_<<<blocks, threads>>>(d_src, d_filter, d_dst, sqrt(gaussianFilter.size()), inputImage.cols - 2*size, inputImage.cols);

    // waits until is done
    cudaDeviceSynchronize();

    cudaMemcpy(gaussianFiltered.data, d_dst, (inputImage.cols - 2*size) * (inputImage.rows - 2*size) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_filter);
    cudaFree(d_dst);



    // STEP 2: SOBEL FILTER

    cv::Mat sobelFiltered = cv::Mat(gaussianFiltered.rows - 2, gaussianFiltered.cols - 2, CV_8UC1);

    anglesMap = cv::Mat(gaussianFiltered.rows - 2, gaussianFiltered.cols - 2, CV_32FC1); //AngleMap

    unsigned char* d_src2 = nullptr;
    unsigned char* d_dst2 = nullptr;
    unsigned char* d_angleMap = nullptr;

    cudaMalloc((void**) &d_src2, (gaussianFiltered.cols) * (gaussianFiltered.rows) * sizeof(unsigned char));
    cudaMalloc((void**) &d_dst2, (gaussianFiltered.cols - 2) * (gaussianFiltered.rows - 2) * sizeof(unsigned char));
    cudaMalloc((void**) &d_angleMap, (gaussianFiltered.cols - 2) * (gaussianFiltered.rows - 2) * sizeof(unsigned char));

    cudaMemcpy(d_src2, gaussianFiltered.data, (gaussianFiltered.cols) * (gaussianFiltered.rows) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst2, sobelFiltered.data, (gaussianFiltered.cols - 2) * (gaussianFiltered.rows - 2) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_angleMap, anglesMap.data, (gaussianFiltered.cols - 2) * (gaussianFiltered.rows - 2) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blocks2((gaussianFiltered.cols - 2) / GRIDVAL + 1, (gaussianFiltered.rows - 2) / GRIDVAL + 1);

    _canny_angle_map_<<<blocks2, threads>>>(d_src2, d_dst2, d_angleMap, gaussianFiltered.cols - 2, gaussianFiltered.cols);

    // waits until is done
    cudaDeviceSynchronize();

    cudaMemcpy(sobelFiltered.data, d_dst2, (gaussianFiltered.cols - 2) * (gaussianFiltered.rows - 2) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(anglesMap.data, d_angleMap, (gaussianFiltered.cols - 2) * (gaussianFiltered.rows - 2) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_src2);
    cudaFree(d_dst2);
    cudaFree(d_angleMap);


    // STEP 3: NON-MAXIMUM SUPRRESSION
    cv::Mat nonMaxSuppressed = cv::Mat(sobelFiltered.rows-2, sobelFiltered.cols-2, CV_8UC1);

    unsigned char* d_src3 = nullptr;
    unsigned char* d_angleMap2 = nullptr;
    unsigned char* d_dst3 = nullptr;

    cudaMalloc((void**) &d_src3, (sobelFiltered.cols) * (sobelFiltered.rows) * sizeof(unsigned char));
    cudaMalloc((void**) &d_angleMap2, (sobelFiltered.cols) * (sobelFiltered.rows) * sizeof(unsigned char));
    cudaMalloc((void**) &d_dst3, (sobelFiltered.cols-2) * (sobelFiltered.rows-2) * sizeof(unsigned char));

    cudaMemcpy(d_src3, sobelFiltered.data, (sobelFiltered.cols) * (sobelFiltered.rows) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_angleMap2, sobelFiltered.data, (sobelFiltered.cols) * (sobelFiltered.rows) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst3, nonMaxSuppressed.data, (sobelFiltered.cols-2) * (sobelFiltered.rows-2) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blocks3((sobelFiltered.cols - 2) / GRIDVAL + 1, (sobelFiltered.rows - 2) / GRIDVAL + 1);

    _nonmax_suppression_<<<blocks3, threads>>>(d_src3, d_angleMap2, d_dst3, sobelFiltered.rows-2, sobelFiltered.rows);

    // waits until is done
    cudaDeviceSynchronize();

    cudaMemcpy(nonMaxSuppressed.data, d_dst3, (sobelFiltered.cols-2) * (sobelFiltered.rows-2) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_src3);
    cudaFree(d_angleMap2);
    cudaFree(d_dst3);



    // STEP 4: STRONG EDGES CONCATENATION
    cv::Mat outputImage = cv::Mat(nonMaxSuppressed.rows, nonMaxSuppressed.cols, CV_8UC1);

    unsigned char* d_src4 = nullptr;
    unsigned char* d_dst4 = nullptr;

    cudaMalloc((void**) &d_src4, nonMaxSuppressed.rows*nonMaxSuppressed.cols*sizeof(unsigned char));
    cudaMalloc((void**) &d_dst4, nonMaxSuppressed.rows*nonMaxSuppressed.cols*sizeof(unsigned char));

    cudaMemcpy(d_src4, nonMaxSuppressed.data, nonMaxSuppressed.rows * nonMaxSuppressed.cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst4, outputImage.data, nonMaxSuppressed.rows * nonMaxSuppressed.cols * sizeof(unsigned char), cudaMemcpyHostToDevice);

    _edges_concatenation_<<<blocks3, threads>>>(d_src4, d_dst4, HIGH_THRESHOLD, LOW_THRESHOLD, nonMaxSuppressed.cols);

    // waits until is done
    cudaDeviceSynchronize();

    cudaMemcpy(outputImage.data, d_dst4, nonMaxSuppressed.rows * nonMaxSuppressed.cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_src4);
    cudaFree(d_dst4);

    return outputImage;
}
