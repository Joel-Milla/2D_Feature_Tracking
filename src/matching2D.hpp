#ifndef matching2D_hpp
#define matching2D_hpp

#include <vector>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>

//* KeyPoint detector type (To decide which algorithm to use)
enum class DetectorType {
    SHITOMASI, HARRIS, FAST,BRISK, ORB, AKAZE, SIFT
};

enum class DescriptorType {
    BRISK, BRIEF, ORB, AKAZE, SIFT
};

enum MatcherType {
    MAT_BF, MAT_FLANN
};

enum SelectorType {
    SEL_NN, SEL_KNN
};

// Helper methods of enums
std::string getStringDetectorType(DetectorType detectorType);
std::string getStringDescriptorType(DescriptorType descriptorType);
bool isBinaryDescriptor(DescriptorType descriptorType);

void visualizeImage(const cv::Mat &img, const std::vector<cv::KeyPoint> &keypoints);
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &img);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &img);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DetectorType detectorType);
void detKeypoints(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &img, const bool visualize, const DetectorType detectorType);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, const cv::Mat &img, cv::Mat &descriptors, const DescriptorType descriptorType);
void matchDescriptors(const std::vector<cv::KeyPoint> &kPtsSource, const std::vector<cv::KeyPoint> &kPtsRef, const cv::Mat &descSource, const cv::Mat &descRef, std::vector<cv::DMatch> &matches, const bool binaryDescriptor, const MatcherType matcherType, const SelectorType selectorType, const bool crossCheck = false);
void limitKeyPoints(const DetectorType &detectorType, std::vector<cv::KeyPoint> &keypoints, int maxKeypoints);
void detectImageKeypoints(const cv::Mat imgGray, const bool visualize_keypoints, const DetectorType detectorType, const bool bFocusOnVehicle, const bool limitKpts, const cv::Rect vehicleRect, std::vector<cv::KeyPoint> &keypoints);

#endif /* matching2D_hpp */
