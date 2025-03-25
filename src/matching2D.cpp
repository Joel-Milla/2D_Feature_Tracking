#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <string>
#include "matching2D.hpp"
#include <opencv2/features2d.hpp>


using namespace std;

/**
 * @brief Function that displays an image and its keypoints 
 * 
 * @param img Image to display
 * @param keypoints keypoints to be displayed
 */
void visualizeImage(const cv::Mat &img, const vector<cv::KeyPoint> &keypoints) {
    //* Create new image combining img and keypoints with format
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //* Show image
    string windowName = "Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {

        //...
    }
    extractor->compute(img, keypoints, descriptors);
}

/**
 * @brief Get the Max Value of the region around a point
 * 
 * @param img Original image
 * @param windows_size Size of the numbers that will be consider to find maximum
 * @param point_row The number of row of the original point 
 * @param point_col The number of column of the original point
 * @return int - Maximum number
 */
int getMaxValue(const cv::Mat& img, int windows_size, int point_row, int point_col) {
    int local_maximum = img.at<int>(point_row, point_col);

    std::vector<pair<int,int>> movements = {{-1,-1},{-1,0},
                                                {-1,1},{0,1},
                                                {1,1},{1,0},
                                                {1,-1},{0,-1}};

    for (int count = 1; count <= windows_size; count++) {
        for (const pair<int,int>& movement : movements) {
            int new_indx = point_row + movement.first;
            int new_jndx = point_col + movement.second;

            // Add boundary checks
            if (new_indx < 0 || new_indx >= img.rows || 
                new_jndx < 0 || new_jndx >= img.cols)
                continue;

            int compared_value = img.at<int>(new_indx, new_jndx);
            local_maximum = max(compared_value, local_maximum);
        }
    }
    
    return local_maximum;
}

/**
 * @brief Given an image, calculate keypoints using harris corner detector 
 * 
 * @param keypoints Where the keypoints will be saved
 * @param img Image to calculate the keypoints
 * @param visualize value to visualize or not
 */
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool visualize) {
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered. This replaces the standar deviation that was explained, so the block size defines the number of neighboors that are considered for detecting the keypoints
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd). Can control the size of the sobel operator. But larger makes it better for noise but also makes it less precise. 
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    //* Detect Harris corners and normalize output
    // This code receives dst_notm_scaled which is the size of the original image, where each position has a value of 8bits. If value is brighter, ith has higher probability of being corner
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );

    //* Iterate through the results obtained, dst_norm_scaled that has values between 0-255 of possibility of being a corner
    for( int indx = 0; indx < dst_norm.rows ; indx++ ) {
        for( int jndx = 0; jndx < dst_norm.cols; jndx++ ) {
            //* Validate that the point is above a certain threshold
            if ( dst_norm_scaled.at<uchar>(indx,jndx) > minResponse ) {

                //* If the current value is a local-maximum (meaning, is max value in a window centered around itself), then save it as a keypoint
                int curr_value = dst_norm_scaled.at<int>(indx, jndx);
                int max_local = getMaxValue(dst_norm_scaled, 1, indx, jndx);

                if (curr_value != max_local) continue;

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(jndx, indx);
                newKeyPoint.size = 2 * apertureSize;

                keypoints.push_back(newKeyPoint);
            }
        }
    }
}


/**
 * @brief Detect keypoints in image using the traditional Shi-Thomasi detector
 * 
 * @param keypoints vector where keypoints will be saved
 * @param img Image to calculate the keypoints
 * @param visualize bool variable
 */
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, const cv::Mat &img, const bool visualize) {
    //* Parameters of the algorithm
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    //* Apply corner detection, the vector corners contain all the corners located
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // Convert the corners of type Point to KeyPoint
    for (auto it = corners.begin(); it != corners.end(); ++it) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
}

/**
 * @brief Implement modern keypoint detectors so when receiving parameters, can run differentt keypoint detector algorithms  
 * 
 * @param keypoints where the keypoints will be saved
 * @param img Image that will be used to calculate keypoints
 * @param detectorType Can be FAST, BRISK, ORB, AKAZE, and SIFT
 */
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DetectorType detectorType) {
    cv::Ptr<cv::FeatureDetector> detector;

    //* Depending on input, set the detector algorithm
    switch (detectorType) {
    case FAST:
        detector = cv::FastFeatureDetector::create(30, true, cv::FastFeatureDetector::TYPE_9_16);
        break;
    case BRISK:
        detector = cv::BRISK::create(30, true, cv::FastFeatureDetector::TYPE_9_16);
        break;
    case ORB:
        detector = cv::ORB::create();
        break;
    case AKAZE:
        detector = cv::AKAZE::create();
        break;
    case SIFT:
        detector = cv::SIFT::create();
        break;
    default:
        return;
    }

    //* Update keypoints
    detector->detect(img, keypoints);
}

/**
 * @brief Function in charge of receiving what keypoint detector algorithm want to run, and then call the function accordingly, and show the results  
 * 
 * @param keypoints vector that save the results
 * @param img that is used to calculate keypoints
 * @param visualize boolean variable to know if visualize results
 * @param detectorType is a variable to select algorithm, can be SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT
 */
void detKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool visualize, DetectorType detectorType) {
    switch (detectorType) {

    case SHITOMASI:
        detKeypointsShiTomasi(keypoints, img, visualize);
        break;
    case HARRIS:
        detKeypointsHarris(keypoints, img, visualize);
        break;
    default:
        detKeypointsModern(keypoints, img, detectorType);
      break;
    }

    if (visualize) visualizeImage(img, keypoints);
}