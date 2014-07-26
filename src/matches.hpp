#ifndef MATCHES_HPP
#define MATCHES_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

void getMatchedPoints(std::vector<cv::Point2f> &lastPoints, std::vector<cv::Point2f> &currentPoints, const std::vector<cv::KeyPoint> &keyPointsLast, const std::vector<cv::KeyPoint> &keyPointsCurrent, const std::vector<cv::DMatch> &matches, float maxDistance);

void drawMatchedFlow(const cv::Mat &currentFrame, cv::Mat &imgMatches, const std::vector<cv::Point2f> &lastPoints, const std::vector<cv::Point2f> &currentPoints, const std::vector<uchar> &inliers);

#endif
