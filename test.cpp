//
// Created by Chu-Hsuan Lin on 2022/2/24.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "segmentation.h"
#include "features.h"

using namespace std;
using namespace cv;

int main5(){


    cv::Mat in, out;
    in.create(2,4,CV_32FC1);
    in.at<float>(0,0) = 5;
    in.at<float>(1,0) = 100;
    in.at<float>(0,1) = 5;
    in.at<float>(1,1) = 100;
    in.at<float>(0,2) = 5;
    in.at<float>(1,2) = 100;
    in.at<float>(0,3) = 5;
    in.at<float>(1,3) = 100;

    float ang = 60;

    cv::Mat rot;
    rot.create(2,2,CV_32FC1);

    rot.at<float>(0,0) = cos(ang);
    rot.at<float>(0,1) = -sin(ang);
    rot.at<float>(1,0) = sin(ang);
    rot.at<float>(1,1) = cos(ang);

    out = rot*in;

    vector<float> r = rotate_matrix(5,100, ang);
    //std::cout <<1 << endl;
    //rotate_m(in, out, ang);

    std::cout << r[0] << " " << out.at<float>(0,0) << endl;
    std::cout << r[1] << " " << out.at<float>(0,1) << endl;
}