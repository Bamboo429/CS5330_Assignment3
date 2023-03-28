//
// Created by Chu-Hsuan Lin on 2022/2/25.
//
#include <opencv2/opencv.hpp>
#include "segmentation.h"
#include "features.h"

using namespace std;
using namespace cv;

int binaryImage(cv::Mat &img, cv::Mat &binary){
    cv::Mat hsv_img, staruation, blur, gray_img;
    binary.create(img.rows, img.cols, CV_8UC1);

    // blur image and convert the HSV
    cv::GaussianBlur(img,blur,Size(5,5),0);
    cv::cvtColor(blur,hsv_img,COLOR_BGR2HSV);
    cv::cvtColor(blur, gray_img, COLOR_BGR2GRAY);

    // to binary image
    for (int i=0; i< img.rows; i++){
        for (int j=0; j< img.cols; j++){
            //set the threshold
            if (hsv_img.at<Vec3b> (i,j)[1] > 100 or hsv_img.at<Vec3b> (i,j)[2] < 100 )
                binary.at<uchar>(i,j) = 255;
            else
                binary.at<Vec3b>(i,j) = 0;
        }
    }

    return 0;

}


// erosion image
void Erosion(cv::Mat &src, cv::Mat &erosion_dst, int erosion_elem , int erosion_size )
{
    // decide different connection method
    int erosion_type = 0;
    if( erosion_elem == 0 )
        erosion_type = MORPH_RECT;  // 8-connect
    else if( erosion_elem == 1 )
        erosion_type = MORPH_CROSS;  //4-connect
    else if( erosion_elem == 2)
        erosion_type = MORPH_ELLIPSE; // ellipse

    Mat element = getStructuringElement( erosion_type,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );
    erode( src, erosion_dst, element );
    //imshow( "Erosion Demo", erosion_dst );
}


void Dilation( cv::Mat &src, cv::Mat &dilation_dst, int dilation_elem , int dilation_size )
{
    // decide different connection method
    int dilation_type = 0;
    if( dilation_elem == 0 )
        dilation_type = MORPH_RECT;  // 8-connect
    else if( dilation_elem == 1 )
        dilation_type = MORPH_CROSS;  //4-connect
    else if( dilation_elem == 2)
        dilation_type = MORPH_ELLIPSE; // ellipse

    Mat element = getStructuringElement( dilation_type,
                                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         Point( dilation_size, dilation_size ) );
    dilate( src, dilation_dst, element );
    //imshow( "Dilation Demo", dilation_dst );
}

void drawRotateBound(cv::Mat &img, RotatedBounding r_bound, Point p_ori){

    // set 4 points of orientated bounding box
    float x_1 = r_bound.p1[0] + p_ori.x;
    float y_1 = r_bound.p1[1] + p_ori.y;
    float x_2 = r_bound.p2[0] + p_ori.x;
    float y_2 = r_bound.p2[1] + p_ori.y;
    float x_3 = r_bound.p3[0] + p_ori.x;
    float y_3 = r_bound.p3[1] + p_ori.y;
    float x_4 = r_bound.p4[0] + p_ori.x;
    float y_4 = r_bound.p4[1] + p_ori.y;

    //draw line
    cv::line(img, Point(x_1,y_1), Point(x_4, y_4), {255, 255, 255}, 3);
    cv::line(img, Point(x_1,y_1), Point(x_3, y_3), {255, 255, 255}, 3);
    cv::line(img, Point(x_2,y_2), Point(x_3, y_3), {255, 255, 255}, 3);
    cv::line(img, Point(x_2,y_2), Point(x_4, y_4), {255, 255, 255}, 3);

}

void drawCentroid(cv::Mat &img, MMoments m, Point p_ori){

    Point p(m.xc+p_ori.x, m.yc+p_ori.y);
    Point p1,p2;

    // define points based on the angle
    p1.x =  (int)round(p.x - 100 * cos(m.ang ));
    p1.y =  (int)round(p.y - 100 * sin(m.ang ));
    p2.x =  (int)round(p.x + 100 * cos(m.ang ));
    p2.y =  (int)round(p.y + 100 * sin(m.ang ));

    //draw line and circle
    cv::line(img, p1,p2, {255,255,255}, 3);
    cv::circle(img, p, 3, {255, 255, 255});

}



