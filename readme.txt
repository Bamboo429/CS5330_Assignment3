//
// Created by Chu-Hsuan Lin on 2022/3/3.
//

=== Project3 - Real-time Object 2-D Recognition ===
https://wiki.khoury.northeastern.edu/display/~chuhsuanlin/Project+3%3A+Real-time+Object+2-D+Recognition

#Built With
Operating system: MacOS Monterey (12.1)
IDE: CLion  https://www.jetbrains.com/clion/

#Installation and Setup
1. Install openCV
2. Modify the CMakeLists.txt
    find_package(OpenCV)
    include_directories(${/usr/local/include/}) //location of your OpenCV
3. Set up a camera, I use EpoCam here.
    https://www.elgato.com/en/epoccam

# Files in the project
1. main.cpp - main file control the progress
2. segmentation.cpp - covert image to binary and different regions
3. feature.cpp - collect all features and cal oriented bounding box
4. classify.cpp - different classifiers and matching methods
5. csv_util.cpp - read and write csv file

#Instructions for running the executables
1. Run the main.cpp file and you will see video from your camera
2. Press key for different functions
    (1) key 'd' : draw orientated bounding box and axis on the label image
    (2) key 's' : save the original image, binary image, morphology image, and labeled image.
    (3) key 'n' : collect training data
        After place the new object, press key 'n' and input the label name in the command line.
        The feature vectors will save in the "features.csv" file. One object at a time.
    (4) key 'c' : run the nearest classifier
        The object name will show on the label image with the same color of object.
    (5) key 'k' : run the KNN classifier
        The object name will show on the label image with the same color of object.
        Also show "unknown" on the label image if the distance value is above a certain threshold.
    (6) key 'o' : show segmentation result, no orientated bounding box, axis, and label on the image.
    (7) key 'q' : exit the program