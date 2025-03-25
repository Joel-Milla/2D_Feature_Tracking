#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <deque>

#include "dataStructures.h"
#include "matching2D.hpp"


using namespace std;

void limitKeyPoints(const DetectorType &detectorType, vector<cv::KeyPoint> &keypoints, int maxKeypoints) {
    //* Limit number of keypoints (helpful for debugging and learning)    
    if (detectorType == DetectorType::SHITOMASI) { 
        // there is no response info, so keep the first 50 as they are sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
    }
    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
    cout << " NOTE: Keypoints have been limited!" << endl;
}

int main(int argc, const char *argv[]) {
    
    //* Set up image reading parameters
    string dataPath = "../";
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";

    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    //* Set up ring buffer
    // If you will have thousands of images, you cannot have a single vector that will hold all the images. Overtime will fill memory and slow program.
    // To solve this, create a data structure to hold limit number of images, and delete the old ones when adding new ones
    int dataBufferSizeLimit = 2;       // no. of images which are held in memory (ring buffer) at the same time
    deque<DataFrame> dataBuffer; // Dequeue for fast insertion/deletion at the ends of the list
    bool bVis = false;            // visualize results

    //* Loop over all imges
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {
        //* Load image and save it into data structure
        // Get the full name of the image by using correct pattern
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // Convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        // Push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_front(frame);

        // Delete oldest image if size limit has been surpassed
        if (dataBuffer.size() > dataBufferSizeLimit) dataBuffer.pop_back();

        //* Detect image keypoints
        // Extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints;
        DetectorType detectorType = DetectorType::SIFT;
        bool visualize_keypoints = false;

        // Obtain time to get keypoints
        auto start = chrono::high_resolution_clock::now();

        // Keypoints array gets added the keypoints
        detKeypoints(keypoints, imgGray, visualize_keypoints, detectorType);

        auto end = chrono::high_resolution_clock::now();
        auto time = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << getStringDetectorType(detectorType) << " detection with n= " << keypoints.size() << " in " << time.count() * 0.001 << " s" << endl;
        
        //* Only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle) {
            // If a keypoint is outside the box parameter, then remove it from the vector
            keypoints.erase(
                std::remove_if(
                    keypoints.begin(),
                    keypoints.end(),
                    [&vehicleRect](cv::KeyPoint const &keyPoint) {
                        int x = keyPoint.pt.x;
                        int y = keyPoint.pt.y;
                        cv::Point p(x, y);
                        return !vehicleRect.contains(p);
                    }
                ),
                keypoints.end()
            );
        }

        // visualizeImage(imgGray, keypoints);

        //* Only use for midterm, for final project must be false
        bool bLimitKpts = false;
        if (bLimitKpts) {
            limitKeyPoints(detectorType, keypoints, 50);
        }
        
        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer.front().keypoints = keypoints;
        

        //* Extract KeyPoint descriptors
        cv::Mat descriptors;
        DescriptorType descriptorType = DescriptorType::BRISK; // BRIEF, ORB, AKAZE, SIFT
        descKeypoints(dataBuffer.front().keypoints, dataBuffer.front().cameraImg, descriptors, descriptorType);

        // Push descriptors for current frame to image just added
        dataBuffer.front().descriptors = descriptors;

        //* Until there are at least two images, then make the match
        if (dataBuffer.size() > 1) { 
            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bool visualize_keypoint_matches = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    return 0;
}