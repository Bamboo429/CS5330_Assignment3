//
// Created by Chu-Hsuan Lin on 2022/2/24.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "segmentation.h"
#include "features.h"
#include "csv_util.h"
#include "classify.h"

using namespace std;
using namespace cv;


bool compareInterval(char* i1, char* i2)
{
    if (strcmp(i1,i2) == 0)
        return 1;
    else
        return 0;
}


int main2(){

    MMoments mm;
    char dir[256] = "/Users/chuhsuanlin/Documents/NEU/Course/Spring 2022/CS 5330 Pattern Reginition & Computer Vission/Project/Project 3/Proj03Examples/comb.jpg";
    cv::Mat img = cv::imread(dir);

    vector<char *> object_names,header;

    vector<vector<float>> features_matrix;
    read_image_data_csv("feature.csv", object_names, features_matrix, header, 0);

    /*
    sort(object_names.begin(), object_names.end(), compareInterval );
    int uniqueClass = std::unique(object_names.begin(), object_names.end(),compareInterval) - object_names.begin();
    std::cout << uniqueClass << endl;
    */


    for(int i=0; i< object_names.size(); i++){
        std::cout << "s " << object_names[i] << endl;
    }

    cv::Mat hsv_img, binary,staruation, blur, gray_img,erosion_binary,dilation_binary,labeled, bounded,rotated_img;
    binary.create(img.rows, img.cols, CV_8UC1);
    labeled.create(img.rows, img.cols, CV_8UC3);

    cv::GaussianBlur(img,blur,Size(5,5),0);


    cv::cvtColor(blur,hsv_img,COLOR_BGR2HSV);
    cv::cvtColor(blur, gray_img, COLOR_BGR2GRAY);

    for (int i=0; i< img.rows; i++){
        for (int j=0; j< img.cols; j++){
            //staruation.at<uchar>(i,j) = hsv_img.at<Vec3b>(i,j)[1];
            if (hsv_img.at<Vec3b> (i,j)[1] > 100 or hsv_img.at<Vec3b> (i,j)[2] < 100)
                binary.at<uchar>(i,j) = 255;
            else
                binary.at<Vec3b>(i,j) = 0;
        }
    }

    binary.copyTo(dilation_binary);
    for(int i=0; i<5; i++) {
        Dilation(dilation_binary, dilation_binary, 0, 1);
    }
    //Dilation( dilation_binary, dilation_binary, 0 , 2);
    for(int i=0; i<5; i++) {
        Erosion(dilation_binary, erosion_binary, 0, 1);
    }
    //Erosion(erosion_binary, erosion_binary, 0, 2);

    Mat labels;
    Mat stats;
    Mat centroids;
    cv::connectedComponentsWithStats(erosion_binary, labels, stats, centroids);

    vector<Vec3b> label_color = {{0,0,255}, {0,255,0}, {255,0,0}, {125,125,125}};
    std::cout << "img.size()=" << img.size() << std::endl;
    std::cout << "labels.size()=" << labels.type() << std::endl;
    std::cout << "centroids.size()=" << centroids.size() << std::endl;
    //std::cout << "stats.cols=" << stats.cols << std::endl;

    //vector<MMoments> moments;
    cv::Mat rotated, r_bounded;
    RotatedBounding r_box;

    rotated.create(2*labeled.rows,2*labeled.cols, labeled.type());
    MMoments m;
    Point up(img.cols,img.rows),down(0,0),left(img.cols,img.rows),right(0,0);

    float y_min = 10000;
    float y_max = -10000;
    float x_min = 10000;
    float x_max = -10000;
    int c=-1;
    for (int l=1; l<stats.rows ;l++){


        int x = stats.at<int>(Point(0, l));
        int y = stats.at<int>(Point(1, l));
        int w = stats.at<int>(Point(2, l));
        int h = stats.at<int>(Point(3, l));

        if (x==0 or y ==0 or x+w == img.cols or y+h == img.rows or w*h < 5000){
            std::cout << "label " << l << " object touch image side" << endl;
        }
        else {
            c+=1;
            for (int i = 0; i < labels.rows; i++) {
                for (int j = 0; j < labels.cols; j++) {
                    if (labels.at<int>(i, j) == l) {
                        labeled.at<Vec3b>(i, j) = label_color[c];
                    }

                }
            }



            Rect rect(x,y,w,h);
            bounded = labels(rect);


            m = MMoments();
            std::cout << "label object " << l <<" now" << endl;
            m = getMoments(bounded,l);
            m.xc += x;
            m.yc += y;

            std::cout << 1 << endl;
            //std::cout << m.ang/3.14*180 << "ang/" << endl;
            c += 1;

            Point p(m.xc, m.yc);
            Point p1,p2;


            p1.x =  (int)round(p.x - 100 * cos(m.ang ));
            p1.y =  (int)round(p.y - 100 * sin(m.ang ));

            p2.x =  (int)round(p.x + 100 * cos(m.ang ));
            p2.y =  (int)round(p.y + 100 * sin(m.ang ));

            cv::line(labeled, p1,p2, {255,255,255}, 3);
            //cv::circle(labeled, p, 3, {255, 255, 255});
            std::cout << 2 << endl;
            //cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
            // adjust transformation matrix

            //cv::Mat mm_bound = labels(rect);
            //Moments mom;
            //mom = moments(mm_bound, true);
            //float ang = 0.5* atan2(mom.nu11, (mom.nu20-mom.nu02));

            //std::cout << 0.5* atan2(mom.nu11, (mom.nu20-mom.nu02))/3.14*180 << " ang" << endl;

            //RotatedBounding r_bound = rotated_bounding(bounded, m.xc-x, m.yc-y, m.ang,l);
            rotated_bounding(bounded, m.xc, m.yc, m.ang, l, r_bounded, r_box);
            std::cout << 3 << endl;
            float x_1 = r_box.p1[0] + x;
            float y_1 = r_box.p1[1] + y;
            float x_2 = r_box.p2[0] + x;
            float y_2 = r_box.p2[1] + y;
            float x_3 = r_box.p3[0] + x;
            float y_3 = r_box.p3[1] + y;
            float x_4 = r_box.p4[0] + x;
            float y_4 = r_box.p4[1] + y;


            cv::line(labeled, Point(x_1,y_1), Point(x_4, y_4), {255, 255, 255}, 3);
            cv::line(labeled, Point(x_1,y_1), Point(x_3, y_3), {255, 255, 255}, 3);
            cv::line(labeled, Point(x_2,y_2), Point(x_3, y_3), {255, 255, 255}, 3);
            cv::line(labeled, Point(x_2,y_2), Point(x_4, y_4), {255, 255, 255}, 3);

            cv::line(labeled, p1,p2, {255,255,255}, 3);
            cv::circle(labeled, p, 3, {255, 255, 255});
            std::cout << 4 << endl;

            vector<float> input_feature = getFeatures(r_box,r_bounded);
            std::cout << 5 << endl;
            vector<vector<float>> nor_features;
            vector<float> class_std;
            vector<float> para = getNormalizePara(features_matrix, "std");
            nor_features = featureNormalize(features_matrix,para, "std");
            std::cout << 6 << endl;

            std::cout << 7 << endl;
            char* Class = KNN(nor_features, input_feature, object_names, 3, "eud");
            std::cout << Class << endl;
            std::cout << 8 << endl;
            //vector<float> distance = calDistance(input_feature,nor_features, "x");
            //char* classify = findMatching(distance, object_names, 0);
            //std::cout << "class " << classify << endl;

            /*
            for(int i=0; i<features.size(); i++){
                std::cout << features[i] << endl;
            }
             */


        }


    }



    //std::cout << nor_features[0][0] << " fff" <<endl;


    cv::imshow("img",img);

    /*for(int i=0; i<bounded.rows; i++){
        for(int j=0; j<bounded.cols; j++){
            bounded.at<int>(i,j) *=20;

        }
    }
    bounded.convertTo(bounded, CV_8UC1);

    cv::imshow("bounded",bounded);
     */
    cv::imshow("binary",binary);
    cv::imshow("label",labeled);
    //cv::imshow("rotated",rotated);
    waitKey(0);

}

