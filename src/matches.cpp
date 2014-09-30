
#include "matches.hpp"

#include <opencv2/imgproc/imgproc.hpp>

void getMatchedPoints(std::vector<cv::Point2f> &lastPoints, std::vector<cv::Point2f> &currentPoints, const std::vector<cv::KeyPoint> &keyPointsLast, const std::vector<cv::KeyPoint> &keyPointsCurrent, const std::vector<cv::DMatch> &matches, float maxDistance) {

	for (const cv::DMatch &match : matches) {
		if (maxDistance == 0 || match.distance <= maxDistance) {
			lastPoints.push_back(keyPointsLast[match.queryIdx].pt);
			currentPoints.push_back(keyPointsCurrent[match.trainIdx].pt);
		}
	}
}

void drawMatchedFlow(const cv::Mat &currentFrame, cv::Mat &imgMatches, const std::vector<cv::Point2f> &lastPoints, const std::vector<cv::Point2f> &currentPoints, const std::vector<uchar> &inliers) {
	currentFrame.copyTo(imgMatches);

	CV_Assert(lastPoints.size() == currentPoints.size());

	if (!inliers.empty()) {
		CV_Assert(lastPoints.size() == inliers.size());

		for (std::vector<cv::Point2f>::size_type i = 0; i != lastPoints.size(); i++) {
			if (inliers[i]) {
				line(imgMatches, lastPoints[i], currentPoints[i], cv::Scalar(0, 0, 255));
				circle(imgMatches, currentPoints[i], 2, cv::Scalar(0, 0, 255));
			}
		}
	}
}

