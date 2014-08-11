
#include "onaframe.hpp"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;

OnaFrame::OnaFrame(int id, const Mat &image, const Mat &cameraMatrix, const std::vector<float> &distCoeffs): _id(id), _image(image), _cameraMatrix(cameraMatrix), _distCoeffs(distCoeffs) {
}

bool OnaFrame::compute(FeatureDetector &detector, DescriptorExtractor &extractor) {
	CV_Assert(!_image.empty());

	detector.detect(_image, _keyPoints);
	extractor.compute(_image, _keyPoints, _descriptors);

	return true;
}

int OnaFrame::getId() const {
	return _id;
}

Mat OnaFrame::getImage() const {
	return _image;
}

bool OnaFrame::match(WPtr frameToMatchTo, DescriptorMatcher &matcher, float maxDistance) {
	if (SPtr trainFrame = frameToMatchTo.lock()) {
		OnaMatch onaMatch;
		onaMatch.trainFrame = frameToMatchTo;

		std::vector<DMatch> matches;
		matcher.match(_descriptors, trainFrame->_descriptors, matches);

		vector<Point2f> queryPoints, trainPoints;
		for (const DMatch &match : matches) {
			if (maxDistance == 0 || match.distance <= maxDistance) {
				queryPoints.push_back(_keyPoints[match.queryIdx].pt);
				trainPoints.push_back(trainFrame->_keyPoints[match.trainIdx].pt);
				onaMatch.queryIdx.push_back(match.queryIdx);
				onaMatch.trainIdx.push_back(match.trainIdx);
			}
		}

		// undistort and put in normal coordinates
		undistortPoints(queryPoints, onaMatch.queryNormalisedPoints, _cameraMatrix, _distCoeffs);
		undistortPoints(trainPoints, onaMatch.trainNormalisedPoints, trainFrame->_cameraMatrix, trainFrame->_distCoeffs);

		frameMatches[trainFrame->getId()] = onaMatch;
	} else {
		return false;
	}

	return true;
}

Mat OnaFrame::findEssentialMatRansac(int id, double ransacMaxDistance, double ransacConfidence) {
	IdMatchMap::iterator frameMatchIt = frameMatches.find(id);

	if (frameMatchIt != frameMatches.end()) {
		OnaMatch *matchPtr = &(frameMatchIt->second);
		matchPtr->essential = findFundamentalMat(matchPtr->trainNormalisedPoints, matchPtr->queryNormalisedPoints, FM_RANSAC, ransacMaxDistance, ransacConfidence, matchPtr->inliers);
		return matchPtr->essential;
	} else {
#ifndef NDEBUG
		std::cout << "Match to frame with id: " << id << " not found" << std::endl;
#endif
	}

	return Mat();
}
