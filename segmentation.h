//
// Created by Chu-Hsuan Lin on 2022/2/25.
//

#include <opencv2/opencv.hpp>
#include "features.h"
using namespace cv;

#ifndef REAL_TIME_OBJECT_2_D_RECOGNITION_SEGMENTATION_H
#define REAL_TIME_OBJECT_2_D_RECOGNITION_SEGMENTATION_H

int binaryImage(cv::Mat &img, cv::Mat &binary);
void Erosion(cv::Mat &src, cv::Mat &erosion_dst, int erosion_elem , int erosion_size );
void Dilation( cv::Mat &src, cv::Mat &dilation_dst, int erosion_elem , int dilation_size );
void drawRotateBound(cv::Mat &img, RotatedBounding r_bound, Point p_ori);
void drawCentroid(cv::Mat &img, MMoments m, Point p_ori);

#endif //REAL_TIME_OBJECT_2_D_RECOGNITION_SEGMENTATION_H
