#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <stdexcept>
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
    string windowName = "show image";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}


/**
 * @brief Find best matches for keypoints in two camera images based on several matching methods (which are brute force or flann. and nearest neighbor or knearest neighbord) with the option of crossChecking
 * 
 * @param kPtsSource keypoints of source
 * @param kPtsRef keypoints reference
 * @param descSource descriptors source
 * @param descRef descriptors reference
 * @param matches vector that will be populated
 * @param binaryDescriptor is boolean variable to know if the descriptors of keypoints are binary
 * @param matcherType is the matcher algorithm to use (brute-force, flann)
 * @param selectorType is the selector algorithm to use (nearest neighbor, or k nearest neighbor)
 * @param crossCheck boolean value if want to apply crosscheck
 */
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
std::vector<cv::DMatch> &matches, bool binaryDescriptor, MatcherType matcherType, SelectorType selectorType, bool crossCheck) {
    //* configure matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;

    //* Can make BruteForce approach (comparing all the distances between keypoints) or FLANN (which constructs a kdTree and returns the same as BF but more optimized)
    switch (matcherType) {

    // If crossCheck, then only bring the ones where both keypoints match. This doesn't work with kNN because crossCheck + NN brings always the points that always match, and kNN always brings k results
    case MAT_BF: {
        int normType = binaryDescriptor ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        break;
    }
    case MAT_FLANN:
        if (descSource.type() != CV_32F) { 
            // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        //* Implement FLANN matching
        matcher = cv::FlannBasedMatcher::create();
      break;
    }

    //* Can make nearest neighbor (the closest one) or use kNN to avoid selecting matches when both possible results are similar and can get FP
    switch (selectorType) {

    case SEL_NN:
         matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        break;
    case SEL_KNN: {
        if (crossCheck)
            std::invalid_argument("Cannot apply crossCheck and at the same time kNN");
        vector<vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, 2);


        //* Filter matches using descriptor distance ratio test
        float threshold = 0.8;
        int discarded = 0;
        for (int indx = 0; indx < knnMatches.size(); indx++) {
            const vector<cv::DMatch>& match = knnMatches[indx];

            // Get ratio between first and second possible match. If they are almost the same (using threshold to know 'almost') then discard both points to avoid having FP
            float firstDist = match[0].distance;
            float secondDist = match[1].distance;
            float ratio = firstDist / secondDist;
            bool valid_match = ratio < threshold;

            if (valid_match)
                matches.push_back(match[0]);
            else
                discarded++;
            
        }
        break;
    }
    }
}

/**
 * @brief Receiving the type of descriptor algorithm wanting to run, compute the descriptors of the image
 * 
 * @param keypoints that will be described
 * @param img were the keypoints reside
 * @param descriptors container where the values will be saved
 * @param descriptorType which is the algorithm to run (BRISK, BRIEF, ORB, AKAZE, SIFT)
 */
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, DescriptorType descriptorType) {
    //* Select descriptor and execute it
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    switch (descriptorType) {
    case DescriptorType::BRISK: {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        descriptor = cv::BRISK::create(threshold, octaves, patternScale);
        break;
    }
    case DescriptorType::BRIEF:
        descriptor = cv::BRISK::create();
        break;
    case DescriptorType::ORB:
        descriptor = cv::ORB::create();
        break;
    case DescriptorType::AKAZE:
        descriptor = cv::AKAZE::create();
        break;
    case DescriptorType::SIFT:
        descriptor = cv::SIFT::create();
        break;
    }
    
    descriptor->compute(img, keypoints, descriptors);
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
    case DetectorType::FAST:
        detector = cv::FastFeatureDetector::create(30, true, cv::FastFeatureDetector::TYPE_9_16);
        break;
    case DetectorType::BRISK:
        detector = cv::BRISK::create(30, true, cv::FastFeatureDetector::TYPE_9_16);
        break;
    case DetectorType::ORB:
        detector = cv::ORB::create();
        break;
    case DetectorType::AKAZE:
        detector = cv::AKAZE::create();
        break;
    case DetectorType::SIFT:
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

    case DetectorType::SHITOMASI:
        detKeypointsShiTomasi(keypoints, img, visualize);
        break;
    case DetectorType::HARRIS:
        detKeypointsHarris(keypoints, img, visualize);
        break;
    default:
        detKeypointsModern(keypoints, img, detectorType);
      break;
    }

    if (visualize) visualizeImage(img, keypoints);
}

//* Helper functions enum
std::string getStringDetectorType(DetectorType detectorType) {
    switch (detectorType) {
    case DetectorType::SHITOMASI:
        return "SHITOMASI";
    case DetectorType::HARRIS:
        return "HARRIS";
    case DetectorType::FAST:
        return "FAST";
    case DetectorType::BRISK:
        return "BRISK";
    case DetectorType::ORB:
        return "ORB";
    case DetectorType::AKAZE:
        return "AKAZE";
    case DetectorType::SIFT:
        return "SIFT";
    default:
        return "";
    }
}

std::string getStringDescriptorType(DescriptorType descriptorType) {
    switch (descriptorType) {

    case DescriptorType::BRISK:
        return "BRISK";
    case DescriptorType::BRIEF:
        return "BRIEF";
    case DescriptorType::ORB:
        return "ORB";
    case DescriptorType::AKAZE:
        return "AKAZE";
    case DescriptorType::SIFT:
        return "SIFT";
    default:
        return "";
    }
}

bool isBinaryDescriptor(DescriptorType descriptorType) {
    switch (descriptorType) {

    case DescriptorType::BRISK:
        return true;
    case DescriptorType::BRIEF:
        return true;
    case DescriptorType::ORB:
        return true;
    case DescriptorType::AKAZE:
        return true;
    case DescriptorType::SIFT:
        return false;
    default:
        return false;
    }
}