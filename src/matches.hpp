/**
 * \file matches.hpp
 * \author James Duley
 * \version 1.0
 */

#ifndef MATCHES_HPP
#define MATCHES_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

/**
 * Gets that matched points that are less than the maximum distance apart.
 *
 * \param lastPoints Vector to store the points in the last frame.
 * \param currentPoints Vector to store the points in the current frame.
 * \param keyPointsLast The keypoints from the last frame.
 * \param keyPointsCurrent The keypoints from the current frame.
 * \param matches The matches between the two frames
 * \param maxDistance The maximum distance between keypoints to allow.
 */
void getMatchedPoints(std::vector<cv::Point2f> &lastPoints, std::vector<cv::Point2f> &currentPoints, const std::vector<cv::KeyPoint> &keyPointsLast, const std::vector<cv::KeyPoint> &keyPointsCurrent, const std::vector<cv::DMatch> &matches, float maxDistance);

/**
 * Draw the the matched flow from the last frame on the current frame.
 *
 * \param currentFrame The current frame which to use for the background of the image.
 * \param imgMatches The image to store the current frame in and draw the keypoint flow on.
 * \param lastPoints The points in the last frame.
 * \param currentPoints The corresponding points in the current frame.
 * \param inliers The points will be drawn for each one that is not 0.
 */
void drawMatchedFlow(const cv::Mat &currentFrame, cv::Mat &imgMatches, const std::vector<cv::Point2f> &lastPoints, const std::vector<cv::Point2f> &currentPoints, const std::vector<uchar> &inliers);

#endif
