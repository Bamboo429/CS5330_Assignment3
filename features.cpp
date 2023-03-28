//
// Created by Chu-Hsuan Lin on 2022/2/27.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "features.h"

using namespace std;
using namespace cv;


MMoments getMoments(cv::Mat src, int label){

    MMoments m;
    m = MMoments();

    // cal moments
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<int>(i, j) == label) {
                m.m00 += 1;
                m.m10 += j;
                m.m01 += i;
                m.m20 += j^2;
                m.m02 += i^2;
                m.m11 += i*j;
            }
        }

    }

    // cal centroid
    m.xc = m.m10/m.m00;
    m.yc = m.m01/m.m00;


    // second moments
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<int>(i, j) == label) {
                m.u02 += (i-m.yc)*(i-m.yc);
                m.u20 += (j-m.xc)*(j-m.xc);
                m.u11 += (i-m.yc)*(j-m.xc);
            }
        }
    }

    m.u11/=m.m00;
    m.u02/=m.m00;
    m.u20/=m.m00;

    // angle
    m.ang = 0.5*atan2(2*m.u11,(m.u20-m.u02));

    return m;

}

void rotate_m(cv::Mat &in, cv::Mat &out, float ang){

    // convert data type to 32F, because * only can use 32F 64F type
    cv::Mat in_32f;
    in.convertTo(in_32f,CV_32FC1);

    cv::Mat rot;
    rot.create(2,2,CV_32FC1);

    // define the rotaion matrix
    rot.at<float>(0,0) = cos(ang);
    rot.at<float>(0,1) = -sin(ang);
    rot.at<float>(1,0) = sin(ang);
    rot.at<float>(1,1) = cos(ang);

    // apply to the input matrix
    out = rot*in_32f;

}


// first version for rotation
vector<float> rotate_matrix(int x, int y, float ang){

    vector<float> matrix;

    float x_r = std::cos(ang) * x - std::sin(ang) * y ;
    float y_r = std::sin(ang) * x + std::cos(ang) * y ;

    matrix.push_back(x_r);
    matrix.push_back(y_r);

    return matrix;
}


int rotated_bounding(cv::Mat labeled, int xc, int yc, float ang, int label, cv::Mat &r_bounded, RotatedBounding &bounding_box){

    // set initiale value
    float y_min = std::numeric_limits<float>::infinity();
    float y_max = -std::numeric_limits<float>::infinity();
    float x_min = std::numeric_limits<float>::infinity();
    float x_max = -std::numeric_limits<float>::infinity();

    cv::Mat in,out,bounding,o_bounding;
    in.create(2,1, CV_32FC1);

    Vec3b back = {0,0,0};
    for (int i=0; i<labeled.rows;i++){
        for(int j=0; j< labeled.cols;j++){
            int x_ = j-xc;
            int y_ = i-yc;

            // rotate degree make image parallel to the major axis
            in.at<float>(0,0) = x_;
            in.at<float>(1,0) = y_;
            rotate_m(in, out, -ang);

            //vector<float> r = rotate_matrix(x_,y_, -ang);
            int x_r = out.at<float>(0,0); //r[0];
            int y_r = out.at<float>(1,0);//r[1];

            // find the min and max of x-axis and y-axis
            if (labeled.at<int>(i, j) == label ){

                if (y_r<y_min) {
                    y_min = y_r;
                }
                if (y_r>y_max) {
                    y_max = y_r;
                }
                if (x_r<x_min) {
                    x_min = x_r;
                }
                if (x_r>x_max) {
                    x_max = x_r;
                }
            }

        }
    }


    // bounding box size
    bounding_box.weight = x_max-x_min;
    bounding_box.height = y_max-y_min;

    // set 4 points of bounding box
    bounding.create(2,4,CV_32FC1);
    bounding.at<float>(0,0) = x_max;
    bounding.at<float>(1,0) = y_max;
    bounding.at<float>(0,1) = x_min;
    bounding.at<float>(1,1) = y_min;
    bounding.at<float>(0,2) = x_min;
    bounding.at<float>(1,2) = y_max;
    bounding.at<float>(0,3) = x_max;
    bounding.at<float>(1,3) = y_min;

    // rotate back to original coordinate
    rotate_m(bounding, o_bounding, ang);

    // shift bounding box
    bounding_box.p1 = {o_bounding.at<float>(0,0)+xc,o_bounding.at<float>(1,0)+yc};
    bounding_box.p2 = {o_bounding.at<float>(0,1)+xc,o_bounding.at<float>(1,1)+yc};
    bounding_box.p3 = {o_bounding.at<float>(0,2)+xc,o_bounding.at<float>(1,2)+yc};
    bounding_box.p4 = {o_bounding.at<float>(0,3)+xc,o_bounding.at<float>(1,3)+yc};

    /* for test
    cv::Mat bounding_index, o_bounding_index;
    bounding_index.create(2,bounding_box.weight*bounding_box.height, CV_32FC1);
    //std::cout << "!!!" << endl;
    int b=0;
    for (int i= x_min; i<x_max; i++){
        for(int j=y_min; j<y_max;j++){

            bounding_index.at<float>(0,b) = i;
            bounding_index.at<float>(1,b) = j;
            b+=1;

        }
    }
    rotate_m(bounding_index, o_bounding_index, ang);
    //std::cout << x_min << " " << y_min << " " << bounding_box.weight << " " << bounding_box.height << endl;
    //Rect rect(x_min,y_min,bounding_box.weight,bounding_box.height);
    //r_bounded = labeled(rect);
     */

    return 0;
}


// cal the filled pixels in the orientated bounding box
float percentfilled(RotatedBounding bounding_box, cv::Mat binaryImage){

    float sum = 0;

    for(int i=0; i<binaryImage.rows; i++){
        for(int j=0; j<binaryImage.cols; j++){
            if (binaryImage.at<int>(i,j) != 0) // not the background
                sum += 1;
        }
    }

    // cal proportion
    float percent = sum/(bounding_box.height*bounding_box.weight);

    return percent;
}

float boundingratio(RotatedBounding bounding_box){

    // ratio height/weight
    return bounding_box.height/bounding_box.weight;
}

vector<double> huMoment(cv::Mat binaryImage){

    // cal moments
    Moments m;
    m = moments(binaryImage, true);

    // hu moments
    double huMoments[7];
    HuMoments(m, huMoments);

    // log transform
    vector<double> hu;
    for(int i = 0; i < 7; i++){
        hu.push_back(-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])));
    }

    return hu;
}

vector<float> getFeatures(RotatedBounding bounding_box, cv::Mat binaryImage){

    vector<float> features;

    // proportion
    float percent = percentfilled(bounding_box,binaryImage);
    features.push_back(percent);

    // ratio
    float ratio = boundingratio(bounding_box);
    features.push_back(ratio);

    // Hu moments
    vector<double> hu = huMoment(binaryImage);
    features.insert(features.end(), hu.begin(), hu.end());

    return features;

}