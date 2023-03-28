//
// Created by Chu-Hsuan Lin on 2022/2/28.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>

using namespace std;
using namespace cv;


float euclidean_distance(vector<float> a, vector<float> b){

    if(a.size()!=b.size()){
        std::cout << "vector size mismatch" << endl;
        return -1;
    }

    int size = a.size();
    float diff =0;
    //cal the difference
    for(int i=0;i<size;i++){
        diff += (a.at(i)-b.at(i))*(a.at(i)-b.at(i));
    }

    //sqrt
    float dis = sqrt(diff);

    return dis;

}


vector<float> getNormalizePara(vector<vector<float>> features, char* method){

    float sum, mean;
    vector<float> feature, vector_std;

    // choose different normalization method
    if (strcmp(method, "std") ==0) {
        for (int j = 0; j < features[0].size(); j++) {
            sum = 0;
            // cal the mean of each feature
            for (int i = 0; i < features.size(); i++) {
                feature.push_back(features[i][j]);
                sum += features[i][j];
            }
            mean = sum / features.size(); //cal mean of each feature

            // cal std of each feature
            float sq_sum = 0;
            for (int i = 0; i < features.size(); i++) {
                sq_sum += (features[i][j] - mean) * (features[i][j] - mean);
            }
            sq_sum /= features.size();
            float std = sqrt(sq_sum);

            //save data
            vector_std.push_back(std);
        }
        return vector_std;
    }
}

vector<vector<float>> featureNormalize (vector<vector<float>> features,  vector<float> para, char* method){

    // initialize normalize features matrix
    vector<vector<float>> nor_features(features.size(),vector<float>(features[0].size()));

    // different nomalization method
    if (strcmp(method, "std") ==0) {
        for (int j = 0; j < features[0].size(); j++) {
            for (int i = 0; i < features.size(); i++) {

                //std normalization
                nor_features[i][j] = features[i][j] / para[j];
            }
        }
        return nor_features;
    }
}


vector<float> calDistance(vector<float> src_feature, vector<vector<float>> features, char* distance_metric) {

    vector<float> feature,diff;

    for (int i = 0; i < features.size(); i++) {
        feature = features[i];
        // euclidean_distance
        if(strcmp(distance_metric, "eud") == 0)
            diff.push_back(euclidean_distance(feature, src_feature));
    }

    return diff;

}



// find index after sorting
//https://www.codegrepper.com/code-examples/cpp/c%2B%2B+sorting+and+keeping+track+of+indexes
template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values
    stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

char* findMatching(vector<float> src_distance, vector<char*> object_names, int threshold){

    vector<char*> match_files, objectnames_copy;


    objectnames_copy.insert(objectnames_copy.begin(), object_names.begin(), object_names.end());

    // sort difference value
    vector<size_t> sort_index = sort_indexes(src_distance);

    //find matching filename according to sort index
    // i = 1 find the best matching
    for (int i=0; i<1; i++){
        match_files.push_back(objectnames_copy[sort_index.at(i)]);
        //std::cout << src_distance[sort_index.at(i)] << endl;
    }
    return match_files[0];
}


bool string_compare(char *a, char *b)
{
    if (strcmp(a,b) == 0)
        return true;
    else
        return false;
}

char* KNN(vector<vector<float>> features, vector<float> src_feature, vector<char*> object_names, int K, char* distance_metric){

    vector<float> feature,src_distance;
    vector<char*> objectnames_copy;

    // duplicate object names for sort
    objectnames_copy.insert(objectnames_copy.begin(), object_names.begin(), object_names.end());
    sort(object_names.begin(), object_names.end(), string_compare);

    //find how many different objects
    int uniqueClass = std::unique(object_names.begin(), object_names.end(),string_compare) - object_names.begin();
    vector<int> countClass(uniqueClass);

    // cal distance between two objects
    src_distance = calDistance(src_feature, features, distance_metric);

    // sort distance
    vector<size_t> sort_index = sort_indexes(src_distance);
    sort(src_distance.begin(), src_distance.end());

    // check the nearest K object belong to which class
    float dis_sum = 0;
    for (int i=0; i<K; i++){
        dis_sum += src_distance[i]/src_feature.size(); // sum distance for recognize unknown object
        for (int j=0; j< uniqueClass; j++){
            //std::cout << "dis " << src_distance[i] << endl;
            if( strcmp(objectnames_copy[sort_index.at(i)],object_names[j])==0){
                countClass[j] +=1;
            }
        }
    }

    // find the maximum number of count
    int maxElementIndex = std::max_element(countClass.begin(),countClass.end()) - countClass.begin();

    // find undefined object, threshold set as 0.1
    if ((dis_sum/K)>0.1){
        return "Unknown" ;
    }
    else {
        return object_names[maxElementIndex];
    }

}


