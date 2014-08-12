
#include "onaframe.hpp"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "pose.hpp"
#include "matches.hpp"

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

		for (const DMatch &match : matches) {
			if (maxDistance == 0 || match.distance <= maxDistance) {
				onaMatch.queryPoints.push_back(_keyPoints[match.queryIdx].pt);
				onaMatch.trainPoints.push_back(trainFrame->_keyPoints[match.trainIdx].pt);
				onaMatch.queryIdx.push_back(match.queryIdx);
				onaMatch.trainIdx.push_back(match.trainIdx);
			}
		}

		// undistort and put in normal coordinates
		undistortPoints(onaMatch.queryPoints, onaMatch.queryNormalisedPoints, _cameraMatrix, _distCoeffs);
		undistortPoints(onaMatch.trainPoints, onaMatch.trainNormalisedPoints, trainFrame->_cameraMatrix, trainFrame->_distCoeffs);

		frameMatches[trainFrame->getId()] = onaMatch;
	} else {
		return false;
	}

	return true;
}

Mat OnaFrame::findEssentialMatRansac(int id, double ransacMaxDistance, double ransacConfidence) {
	Mat E;

	if (OnaMatch *matchPtr = getMatchById(id)) {
		setEssentialMatRansac(*matchPtr, ransacMaxDistance, ransacConfidence);
		matchPtr->essential.copyTo(E);
	}

	return E;
}

OnaFrame::Pose OnaFrame::findPoseDiff(int id) {
	Pose poseDiff;

	if (OnaMatch *matchPtr = getMatchById(id)) {
		setPoseDiff(*matchPtr);
		poseDiff =  matchPtr->poseDiff;
	}

	return poseDiff;
}

Mat OnaFrame::drawMatchedFlowFrom(int id) {
	Mat image;

	if (OnaMatch *matchPtr = getMatchById(id)) {
		drawMatchedFlow(_image, image, matchPtr->trainPoints, matchPtr->queryPoints, matchPtr->inliers);
	}

	return image;
}

OnaFrame::OnaMatch *OnaFrame::getMatchById(int id) {
	IdMatchMap::iterator frameMatchIt = frameMatches.find(id);

	if (frameMatchIt != frameMatches.end()) {
		return &(frameMatchIt->second);
#ifndef NDEBUG
	} else {
		std::cout << "Match to frame with id: " << id << " not found" << std::endl;
#endif
	}

	return nullptr;
}

void OnaFrame::setEssentialMatRansac(OnaMatch &match, double ransacMaxDistance, double ransacConfidence) {
	match.essential = findFundamentalMat(match.trainNormalisedPoints, match.queryNormalisedPoints, FM_RANSAC, ransacMaxDistance, ransacConfidence, match.inliers);
}

void OnaFrame::setPoseDiff(OnaMatch &match) {
	recoverPose(match.essential, match.trainNormalisedPoints, match.queryNormalisedPoints, match.poseDiff.R, match.poseDiff.t, noArray());
}
