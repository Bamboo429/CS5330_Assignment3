#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "segmentation.h"
#include "features.h"
#include "csv_util.h"
#include "classify.h"

using namespace std;
using namespace cv;

// set color for different objects
vector<Vec3b> label_color = {{0,0,255}, {0,255,0}, {255,0,0}, {255,255,0}, {96,164,244}, {255,204,204}};

int main(int argc, char *argv[]) {
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(0);
    if( !capdev->isOpened() ) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    // set image size for quick computation
    capdev->set(cv::CAP_PROP_FRAME_WIDTH,720);
    capdev->set(cv::CAP_PROP_FRAME_HEIGHT,1280);
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);

    cv::namedWindow("Video", 1); // identifies a window


    cv::Mat frame, binary, dilation_binary, morphology_binary;
    cv::Mat labels,stats,centroids;
    cv::Mat labeled;

    RotatedBounding r_box;
    cv::Mat bounded, r_bounded;
    vector<float> input_feature;

    //set flag
    bool flag_c = false; // classify mode
    bool flag_k = false; // KNN classifier
    bool flag_d = false;

    // read database
    vector<char *> object_names,header;
    vector<vector<float>> features_matrix;
    vector<vector<float>> nor_features, input_features;
    vector<float> class_std;

    // if no csv file for features, show only segmentation image and bounding box
    //if(read_image_data_csv("feature.csv", object_names, features_matrix, header, 0)==-1){
    //    std::cout << "No database exist " << endl;
    //    std::cout << "Press 'n' to get training data " << endl;
    //}

    for(;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        labeled =  cv::Mat::zeros(frame.size(), CV_8UC3);

        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }

        // use threshold to generate binary image
        binaryImage(frame, binary);

        // do dilation and erosion to fill hole and thin
        binary.copyTo(morphology_binary);
        for(int i=0; i<5; i++) {
            Dilation(morphology_binary, morphology_binary, 0, 1);
        }
        //Dilation( dilation_binary, dilation_binary, 0 , 2);
        for(int i=0; i<5; i++) {
            Erosion(morphology_binary, morphology_binary, 0, 1);
        }


        // region growing for segmentation
        cv::connectedComponentsWithStats(morphology_binary, labels, stats, centroids);

        int c=-1; // color control

        for (int l=1; l<stats.rows ;l++) {

            // get bounding box of different label
            int x = stats.at<int>(Point(0, l));
            int y = stats.at<int>(Point(1, l));
            int w = stats.at<int>(Point(2, l));
            int h = stats.at<int>(Point(3, l));

            // filter bounding box touch edge and region smaller than threshold
            if (x == 0 or y == 0 or x + w == frame.cols or y + h == frame.rows or w * h < 2000) {
                //std::cout << "label " << l << " object touch image side" << endl;
            }
            else {
                //std::cout << "label object " << l << endl;

                c += 1; // set color
                // color different object based on labels from segmentation
                for (int i = 0; i < labels.rows; i++) {
                    for (int j = 0; j < labels.cols; j++) {
                        if (labels.at<int>(i, j) == l) {
                            labeled.at<Vec3b>(i, j) = label_color[c];
                        }
                    }
                }

                // get subregion of image
                Rect rect(x,y,w,h);
                //bounded = labels(rect);
                labels(rect).copyTo(bounded);

                for(int i =0; i<bounded.rows;i++){
                    for(int j =0; j<bounded.cols;j++) {
                        if (bounded.at<int>(i, j) != l)
                            bounded.at<int>(i, j) = 0;
                    }
                }


                // cal features moments
                MMoments m = MMoments();
                m = getMoments(bounded, l);


                // find rotated bounding box
                //int rotated_bounding(cv::Mat labeled, int xc, int yc, float ang, int label, cv::Mat &r_bounded, RotatedBounding &bounding_box)
                rotated_bounding(bounded, m.xc, m.yc, m.ang, l, r_bounded, r_box);
                if (flag_d == true) {
                    drawCentroid(labeled, m, Point(x,y)); // draw centroid and axis
                    drawRotateBound(labeled, r_box, Point(x, y)); // draw rotated bounding box
                }

                input_feature = getFeatures(r_box,bounded);
                char* classify;

                // classify mode - nearest-neighbor recognition
                if (flag_c == true) {
                    // collect features and normalize
                    vector<vector<float>> nor_input, input_features;
                    input_features.push_back(input_feature);
                    nor_input = featureNormalize(input_features, class_std, "std"); // input feature

                    // cal distance
                    vector<float> distance = calDistance(nor_input[0],nor_features, "eud");
                    // classifier
                    classify = findMatching(distance, object_names, 0);
                    // put text on image
                    cv::putText(labeled,classify,Point(x,y),cv::FONT_HERSHEY_DUPLEX,1,label_color[c],2,false);

                }

                // KNN classifier
                else if(flag_k == true){

                    // collect features and normalize
                    vector<vector<float>> nor_input, input_features;
                    // collect features and normalize
                    input_features.push_back(input_feature);
                    nor_input = featureNormalize(input_features, class_std, "std");

                    classify = KNN(nor_features, nor_input[0], object_names, 1, "eud");
                    // put text on image
                    cv::putText(labeled,classify,Point(x,y),cv::FONT_HERSHEY_DUPLEX,1,label_color[c],2,false);
                }

            }
        }


        // see if there is a waiting keystroke
        char key = cv::waitKey(10);

        // save images
        if (key == 's') {

            cv::imwrite("ori.jpg", frame);
            cv::imwrite("binary.jpg", binary);
            cv::imwrite("clean.jpg", morphology_binary);
            cv::imwrite("label.jpg", labeled);
        }

        // draw bounding box and axis
        else if(key =='d'){
            flag_d = true;
        }

        // train data
        else if (key == 'n'){

            flag_c = false;

            char label_name[64];
            // get label name
            std::cout << "Please input the label name of the object : ";
            std::cin >> label_name;
            //save feature vector to csv file
            append_image_data_csv("feature.csv", label_name, input_feature, 0);

            //strcat(label_name,".jpg");
            //cv::imwrite(label_name, frame);
        }

        // Nearest classifier
        else if (key == 'c' and flag_c == false){

            object_names.clear();
            features_matrix.clear();

            // read feature vectors from csv file
            if(read_image_data_csv("feature.csv", object_names, features_matrix, header, 0)==-1){

                std::cout << "No database exist " << endl;
                std::cout << "Press 'n' to get training data " << endl;
            }

            else {
                // normalize features
                class_std = getNormalizePara(features_matrix, "std");
                nor_features = featureNormalize(features_matrix, class_std, "std"); // database features
                flag_c = true;
                flag_k = false;
                flag_d = true;

                //cv::putText(labeled, "Classify", Point(0, 0), cv::FONT_HERSHEY_DUPLEX, 1, {255, 255, 255}, 2, false);
            }
            std::cout << "Running Nearest classifier.... " << endl;
            std::cout << object_names.size() << " feature vectors in the CVS file" << endl;

        }

        // KNN classifier
        else if (key == 'k' and flag_k == false){

            object_names.clear();
            features_matrix.clear();

            // read feature vectors from csv file
            if(read_image_data_csv("feature.csv", object_names, features_matrix, header, 0)==-1){
                std::cout << "No database exist " << endl;
                std::cout << "Press 'n' to get training data " << endl;
            }
            else {
                flag_c = false;
                flag_k = true;
                flag_d = true;
                // normalize features
                class_std = getNormalizePara(features_matrix, "std");
                nor_features = featureNormalize(features_matrix, class_std, "std"); // database features
            }
            std::cout << "Running KNN classifier.... " << endl;
            std::cout << object_names.size() << " feature vectors in the CVS file" << endl;

        }

        // only segmentation mode
        else if(key == 'o'){
            flag_d = false;
            flag_c = false;
            flag_k = false;
        }

        else if( key == 'q') {
            break;
        }

        cv::imshow("Video", frame);
        cv::imshow("Binary", binary);
        cv::imshow("Label", labeled);
        cv::imshow("Cleaned_binary", morphology_binary);

    }

    delete capdev;
    return(0);
}
