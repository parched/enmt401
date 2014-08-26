
#include "onaframe.hpp"

#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "pose.hpp"
#include "matches.hpp"

using namespace cv;

/**
 * \brief A match between frames.
 */
struct OnaFrame::OnaMatch {
	WPtr queryFrame;
	WPtr trainFrame;
	std::vector<int> queryIdx, trainIdx;
	std::vector<cv::Point2f> queryPoints, trainPoints;
	std::vector<cv::Point2f> queryNormalisedPoints, trainNormalisedPoints;
	cv::Mat essential;
	std::vector<uchar> inliers;
	Pose poseDiff;
};

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

void OnaFrame::match(SPtr queryFrame, SPtr trainFrame, DescriptorMatcher &matcher, float maxDistance) {
	MatchSPtr onaMatch(new OnaMatch);
	queryFrame->_frameMatchesFrom[trainFrame->_id] = onaMatch;
	trainFrame->_frameMatchesTo[queryFrame->_id] = onaMatch;

	onaMatch->queryFrame = queryFrame;
	onaMatch->trainFrame = trainFrame;

	std::vector<DMatch> matches;
	if (!queryFrame->_descriptors.empty() && !trainFrame->_descriptors.empty()) {
		matcher.match(queryFrame->_descriptors, trainFrame->_descriptors, matches);

		for (const DMatch &match : matches) {
			if (maxDistance == 0 || match.distance <= maxDistance) {
				onaMatch->queryPoints.push_back(queryFrame->_keyPoints[match.queryIdx].pt);
				onaMatch->trainPoints.push_back(trainFrame->_keyPoints[match.trainIdx].pt);
				onaMatch->queryIdx.push_back(match.queryIdx);
				onaMatch->trainIdx.push_back(match.trainIdx);
			}
		}

		if (onaMatch->queryPoints.size() > 0) {
			// undistort and put in normal coordinates
			undistortPoints(onaMatch->queryPoints, onaMatch->queryNormalisedPoints, queryFrame->_cameraMatrix, queryFrame->_distCoeffs);
			undistortPoints(onaMatch->trainPoints, onaMatch->trainNormalisedPoints, trainFrame->_cameraMatrix, trainFrame->_distCoeffs);
		}
	}
}

Mat OnaFrame::findEssentialMatRansac(int id, double ransacMaxDistance, double ransacConfidence) {
	Mat E;

	if (MatchSPtr matchPtr = getMatchById(id)) {
		setEssentialMatRansac(*matchPtr, ransacMaxDistance, ransacConfidence);
		matchPtr->essential.copyTo(E);
	}

	return E;
}

OnaFrame::Pose OnaFrame::findPoseDiff(int id) {
	Pose poseDiff;

	if (MatchSPtr matchPtr = getMatchById(id)) {
		setPoseDiff(*matchPtr);
		poseDiff =  matchPtr->poseDiff;
	}

	return poseDiff;
}

Mat OnaFrame::drawMatchedFlowFrom(int id) {
	Mat image;

	if (MatchSPtr matchPtr = getMatchById(id)) {
		drawMatchedFlow(_image, image, matchPtr->trainPoints, matchPtr->queryPoints, matchPtr->inliers);
	}

	return image;
}

OnaFrame::MatchSPtr OnaFrame::getMatchById(int id) {
	IdMatchMapFrom::iterator frameMatchIt = _frameMatchesFrom.find(id);

	if (frameMatchIt != _frameMatchesFrom.end()) {
		return frameMatchIt->second;
#ifndef NDEBUG
	} else {
		std::cout << "Match to frame with id: " << id << " not found" << std::endl;
#endif
	}

	return nullptr;
}

void OnaFrame::setEssentialMatRansac(OnaMatch &match, double ransacMaxDistance, double ransacConfidence) {
	if (match.queryNormalisedPoints.size() > 0) {
		match.essential = findFundamentalMat(match.trainNormalisedPoints, match.queryNormalisedPoints, FM_RANSAC, ransacMaxDistance, ransacConfidence, match.inliers);
	}
}

void OnaFrame::setPoseDiff(OnaMatch &match) {
	if (match.queryNormalisedPoints.size() > 0) {
		recoverPose(match.essential, match.trainNormalisedPoints, match.queryNormalisedPoints, match.poseDiff.R, match.poseDiff.t, noArray());
	}
}
