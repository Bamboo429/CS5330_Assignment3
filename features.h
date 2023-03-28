//
// Created by Chu-Hsuan Lin on 2022/2/27.
//

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef REAL_TIME_OBJECT_2_D_RECOGNITION_FEATURES_H
#define REAL_TIME_OBJECT_2_D_RECOGNITION_FEATURES_H

class MMoments {       // The class
public:             // Access specifier
    float m00 = 0;
    float m10 = 0;
    float m01 = 0;
    float m20 = 0;
    float m02 = 0;
    float m11 = 0;
    float u11 = 0;
    float u02 = 0;
    float u20 = 0;

    float ang = 0;
    int xc = 0;
    int yc = 0;

};

class RotatedBounding {
public:

    vector<float> p1;
    vector<float> p2;
    vector<float> p3;
    vector<float> p4;

    vector<int> center;

    float weight;
    float height;
    float ang;

};
MMoments getMoments(cv::Mat src, int label);
vector<float> rotate_matrix(int x, int y, float ang);
int rotated_bounding(cv::Mat labeled, int xc, int yc, float ang, int label, cv::Mat &r_bounded, RotatedBounding &bounding_box);
vector<float> getFeatures(RotatedBounding bounding_box, cv::Mat binaryImage);
void rotate_m(cv::Mat &in, cv::Mat &out, float ang);

#endif //REAL_TIME_OBJECT_2_D_RECOGNITION_FEATURES_H
