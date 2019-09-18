//
// Created by marco on 05/09/19.
//

#ifndef EDGEDETECTOR_CUDACANNY_H
#define EDGEDETECTOR_CUDACANNY_H


#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>



class cudaCanny {

public:

    cudaCanny(cv::Mat inputImage, const char *imgName, int size, double sigma);
    cv::Mat computeCuda();

private:
    cv::Mat inputImage;
    const char* inputImageFileName;
    cv::Mat outputImage;

    cv::Mat anglesMap;
    std::vector<double> gaussianFilter;

    void generateFilter(int size, double sigma);
};


#endif //EDGEDETECTOR_CUDACANNY_H
