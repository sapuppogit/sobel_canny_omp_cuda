//
// Created by marco on 22/08/19.
//

#ifndef EDGEDETECTOR_CUDASOBEL_H
#define EDGEDETECTOR_CUDASOBEL_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define KERNEL_DIM 3

class cudaSobel {


public:

    cudaSobel(cv::Mat inImage, const char *imgName);

    void computeCuda();

    void displayOutputImg(const cv::String title);
    const char* getInputImageFileName();
    double getComputationTime();

private:

    cv::Mat inputImage;
    cv::Mat outputImage;
    const char *inputImageFileName;
    double executionTime = 0;

    int norm2(int x, int y) {
        return sqrt(x*x + y*y);
    }

    const int xKernel[KERNEL_DIM][KERNEL_DIM] = {
            {1, 0, -1},
            {2, 0, -2},
            {1, 0, -1}
    };

    const int yKernel[KERNEL_DIM][KERNEL_DIM] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
    };

    struct grad {
        int gradX;
        int gradY;
    };

    grad gradient(int a, int b,int c, int d,int e, int f, int g, int h, int i);
};


#endif //EDGEDETECTOR_CUDASOBEL_H
