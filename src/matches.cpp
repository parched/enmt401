
#include "matches.hpp"

using namespace cv;

void getMatchedPoints(vector<Point2f> &lastPoints, vector<Point2f> &currentPoints, const vector<KeyPoint> &keyPointsLast, const vector<KeyPoint> &keyPointsCurrent, const vector<DMatch> &matches, float maxDistance) {

	for (const DMatch &match : matches) {
		if (maxDistance == 0 || match.distance <= maxDistance) {
			lastPoints.push_back(keyPointsLast[match.queryIdx].pt);
			currentPoints.push_back(keyPointsCurrent[match.trainIdx].pt);
		}
	}
}

void drawMatchedFlow(const Mat &currentFrame, Mat &imgMatches, const vector<Point2f> &lastPoints, const vector<Point2f> &currentPoints, const vector<uchar> &inliers) {
	currentFrame.copyTo(imgMatches);

	assert(lastPoints.size() == currentPoints.size());
	assert(lastPoints.size() == inliers.size());

	for (vector<Point2f>::size_type i = 0; i != lastPoints.size(); i++) {
		if (inliers[i]) {
			line(imgMatches, lastPoints[i], currentPoints[i], Scalar(0, 0, 255));
			circle(imgMatches, currentPoints[i], 2, Scalar(0, 0, 255));
		}
	}
}

