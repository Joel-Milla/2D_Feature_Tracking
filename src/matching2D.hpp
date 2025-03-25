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

// Helper methods of enums
std::string getStringDetectorType(DetectorType detectorType);

void visualizeImage(const cv::Mat &img, const std::vector<cv::KeyPoint> &keypoints);
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DetectorType detectorType);
void detKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool visualize, DetectorType detectorType);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, DescriptorType descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

#endif /* matching2D_hpp */
