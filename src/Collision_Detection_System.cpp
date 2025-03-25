#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <deque>

#include "dataStructures.h"
#include "matching2D.hpp"

/**
 * @brief For testing purposes only, limit the amount of keypoints to visualize and debug the program
 * 
 * @param detectorType 
 * @param keypoints 
 * @param maxKeypoints 
 */
void limitKeyPoints(const DetectorType &detectorType, std::vector<cv::KeyPoint> &keypoints, int maxKeypoints) {
    //* Limit number of keypoints (helpful for debugging and learning)    
    if (detectorType == DetectorType::SHITOMASI) { 
        // there is no response info, so keep the first 50 as they are sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
    }
    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
    std::cout << " NOTE: Keypoints have been limited!" << std::endl;
}

/**
 * @brief Function that shows the matches between keypoints 
 * 
 * @param dataBuffer that contains two images that have their own matches
 * @param matches vector that contains the correspondence between keypoints
 */
void visualizeMatches(const std::deque<DataFrame> &dataBuffer, const std::vector<cv::DMatch> &matches) {
    cv::Mat matchImg = (dataBuffer.front().cameraImg).clone();
    cv::drawMatches(dataBuffer.back().cameraImg, dataBuffer.back().keypoints,
                    dataBuffer.front().cameraImg, dataBuffer.front().keypoints,
                    matches, matchImg,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::string windowName = "Matching keypoints between two camera images";
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    std::cout << "Press key to continue to next image" << std::endl;
    cv::waitKey(0); // wait for key to be pressed
}

/**
 * @brief Get the Image object by receiving the path to it
 * 
 * @param imgStartIndex of the image
 * @param imgEndIndex of the image
 * @param imgFillWidth are the number of zeros need to preprend
 * @param imgIndex of our dataset
 * @param imgPathPrefix the path that goes before the name of image
 * @param imgFileType 'png', etc.
 * @return cv::Mat gray image
 */
cv::Mat getImage(const int imgStartIndex, const int imgEndIndex, const int imgFillWidth, const int imgIndex, const std::string imgPathPrefix, const std::string imgFileType) {
    // Get the full name of the image by using correct pattern
    std::ostringstream imgNumber;
    imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
    std::string imgFullFilename = imgPathPrefix + imgNumber.str() + imgFileType;

    // Convert to grayscale
    cv::Mat img, imgGray;
    img = cv::imread(imgFullFilename);
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    return imgGray;
}

/**
 * @brief Function which obtains the keypoints of an image based on the parameters received 
 * 
 * @param imgGray which is image that we will compute keypoints
 * @param visualize_keypoints bool variable to know if visualize image
 * @param detectorType is the type of detector algorithm to use
 * @param bFocusOnVehicle if true then removes the keypoints that are not part of vehicle in front
 * @param limitKpts for debugging purposes, to limit kpts and see them in image
 * @param vehicleRect rectangle that crops the vehicle in front
 * @param keypoints vector where the keypoints will be saved
 */
void detectImageKeypoints(const cv::Mat imgGray, const bool visualize_keypoints, const DetectorType detectorType, const bool bFocusOnVehicle, const bool limitKpts, const cv::Rect vehicleRect, std::vector<cv::KeyPoint> keypoints) {
    // Obtain time to get keypoints
    auto start = std::chrono::high_resolution_clock::now();

    // Keypoints array gets added the keypoints
    detKeypoints(keypoints, imgGray, visualize_keypoints, detectorType);

    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << getStringDetectorType(detectorType) << " detection with n= " << keypoints.size() << " in " << time.count() * 0.001 << " s" << std::endl;
    
    //* Only keep keypoints on the preceding vehicle
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

    if (visualize_keypoints)
        visualizeImage(imgGray, keypoints);

    //* Only use for debugging purposes when want to visualize the keypoints
    if (limitKpts) {
        limitKeyPoints(detectorType, keypoints, 50);
    }
}

/**
 * @brief Function that loops over the 10 images with the configuration received
 * 
 * @param imgStartIndex start of iamge
 * @param imgEndIndex end index to loop
 * @param imgFillWidth width of prepending zeros
 * @param imgPathPrefix path to image
 * @param imgFileType 'png', etc.
 * @param dataBufferSizeLimit size limit of buffer
 * @param visualize_keypoints bool vairable to control visualziation of points
 * @param visualize_keypoint_matches bool variable to control visualization of keypoint matches
 * @param detectorType type of detector algorithm [Harris, Fast, Brisk, etc]
 * @param descriptorType type of descriptor algorithm [Brisk, Brief, Orb, etc]
 * @param matcherType type of mathcer algorithm [BF, Flann]
 * @param selectorType type of selector algoirhtm [NN, kNN]
 * @param bFocusOnVehicle only focus on vehicle upfront
 * @param dataBuffer unique continer that changes during program, it holds the images, up to limit of dataBufferSizeLimit
 */
void loopOverImages(const int imgStartIndex, const int imgEndIndex, const int imgFillWidth, const std::string imgPathPrefix, const std::string imgFileType, const int dataBufferSizeLimit, const bool visualize_keypoints, const bool visualize_keypoint_matches, const DetectorType detectorType, const DescriptorType descriptorType, const MatcherType matcherType, const SelectorType selectorType, const bool bFocusOnVehicle, std::deque<DataFrame> dataBuffer) {
    //* Loop over all imges
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {
        //* Main variables used across all the function
        // Variables to extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints;
        
        bool limitKpts = false; // Only for debugging purposes
        
        // Variables to get descriptors of keypoints
        cv::Mat descriptors;

        // Bounding box of preceding vehicle
        cv::Rect vehicleRect(535, 180, 180, 150);

        // Variables to get matches between two images
        std::vector<cv::DMatch> matches;
        bool is_des_binary = isBinaryDescriptor(descriptorType); // DES_BINARY, DES_HOG

        //* Start creating and matching keypoints
        std::cout << "Image index: " << imgIndex << std::endl;

        //* Load image and save it into data structure
        cv::Mat imgGray = getImage(imgStartIndex, imgEndIndex, imgFillWidth, imgIndex, imgPathPrefix, imgFileType);

        // Push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_front(frame);

        // Delete oldest image if size limit has been surpassed
        if (dataBuffer.size() > dataBufferSizeLimit) dataBuffer.pop_back();

        //* Detect image keypoints
        detectImageKeypoints(imgGray, visualize_keypoints, detectorType, bFocusOnVehicle, limitKpts, vehicleRect, keypoints);
        
        // Push keypoints for current frame to front of data buffer
        dataBuffer.front().keypoints = keypoints;

        //* Extract KeyPoint descriptors
        descKeypoints(dataBuffer.front().keypoints, dataBuffer.front().cameraImg, descriptors, descriptorType);

        // Push descriptors for current frame to image just added
        dataBuffer.front().descriptors = descriptors;

        //* Until there are at least two images, then make the match
        if (dataBuffer.size() > 1) { 
            matchDescriptors(dataBuffer.back().keypoints, dataBuffer.front().keypoints, dataBuffer.back().descriptors, dataBuffer.front().descriptors, matches, is_des_binary, matcherType, selectorType);

            // Store matches in current data frame
            dataBuffer.front().kptMatches = matches;

            // visualize matches between current and previous image
            if (visualize_keypoint_matches)
                visualizeMatches(dataBuffer, matches);
        }
        std::cout << std::endl;
    } // eof loop over all images
}

int main(int argc, const char *argv[]) {
    
    //* Set up image reading parameters
    std::string dataPath = "../";
    std::string imgBasePath = dataPath + "images/";
    std::string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    std::string imgPathPrefix = imgBasePath + imgPrefix;
    std::string imgFileType = ".png";

    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    //* Set up ring buffer
    // If you have thousands of images, you cannot have a single vector that will hold all the images. Overtime will fill memory and slow program.
    // To solve this, create a data structure to hold limit number of images, and delete the old ones when adding new ones
    int dataBufferSizeLimit = 2;       // no. of images which are held in memory (ring buffer) at the same time

    // Visualization variables
    bool visualize_keypoints = true;
    bool visualize_keypoint_matches = false;

    // Types
    DetectorType detectorType = DetectorType::SIFT;
    DescriptorType descriptorType = DescriptorType::BRISK; // BRIEF, ORB, AKAZE, SIFT
    MatcherType matcherType = MAT_BF;
    SelectorType selectorType = SEL_NN;   // SEL_NN, SEL_KNN

    // Variables to keep keypoints on preceding vehicle
    bool bFocusOnVehicle = true;

    //* The lines below are in charge of looping over the program
    std::deque<DataFrame> dataBuffer; // Dequeue for fast insertion/deletion at the ends of the list
    loopOverImages(imgStartIndex, imgEndIndex, imgFillWidth, imgPathPrefix, imgFileType, dataBufferSizeLimit, visualize_keypoints, visualize_keypoint_matches, detectorType, descriptorType, matcherType, selectorType, bFocusOnVehicle, dataBuffer);

    return 0;
}