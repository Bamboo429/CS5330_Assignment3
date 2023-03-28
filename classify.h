//
// Created by Chu-Hsuan Lin on 2022/2/28.
//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#ifndef REAL_TIME_OBJECT_2_D_RECOGNITION_CLASSIFY_H
#define REAL_TIME_OBJECT_2_D_RECOGNITION_CLASSIFY_H


char* findMatching(vector<float> src_distance, vector<char*> object_names, int threshold);
vector<float> calDistance(vector<float> src_feature, vector<vector<float>> features, char* distance_metric);
vector<float> getNormalizePara(vector<vector<float>> features, char* method);
vector<vector<float>> featureNormalize (vector<vector<float>> features,  vector<float> para, char* method);
char* KNN(vector<vector<float>> features, vector<float> src_feature, vector<char*> object_names, int K, char* distance_metric);

#endif //REAL_TIME_OBJECT_2_D_RECOGNITION_CLASSIFY_H
