/**
 * \file onaframe.cpp
 * \brief An optical navigation aid frame class implementation.
 * \author James Duley
 * \version 1.0
 * \date 2014-08-26
 */

/* Copyright (C) 
 * 2014 - James Duley
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 */

#include "onaframe.hpp"

#include <iostream>
#include <algorithm>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "pose.hpp"
#include "matches.hpp"


struct ona::Frame::OnaMatch {
	WPtr queryFrame;
	WPtr trainFrame;
	std::vector<int> queryIdx, trainIdx;
	std::vector<cv::Point2f> queryPoints, trainPoints;
	std::vector<cv::Point2f> queryNormalisedPoints, trainNormalisedPoints;
	cv::Mat essential;
	std::vector<uchar> inliers;
	Pose poseDiff;
	cv::Mat_<float> points3D;
	float scale = 1.0;

	void setScaleFrom(OnaMatch &matchFrom);
};

void ona::Frame::OnaMatch::setScaleFrom(OnaMatch &matchFrom) {
	std::vector<int> thisMatchIdx, fromMatchIdx;

	auto fromStartIter = matchFrom.queryIdx.begin();
	auto fromEndIter = matchFrom.queryIdx.end();

	for (std::size_t i = 0; i < inliers.size(); i++) {
		if (inliers[i]) {
			auto fromFoundIter = std::find(fromStartIter, fromEndIter, trainIdx[i]);
			if (fromFoundIter != fromEndIter) {
				std::size_t fromIterIdx = fromFoundIter - fromStartIter;
				if (matchFrom.inliers[fromIterIdx]) {
					thisMatchIdx.push_back(i);
					fromMatchIdx.push_back(fromIterIdx);
					fromStartIter = fromFoundIter;
				}
			}
		}
	}


	if (thisMatchIdx.empty()) {
		return;
	} else {
		std::vector<float> scales(thisMatchIdx.size());

		cv::Mat_<float> &fromPts = matchFrom.points3D;
		cv::Mat_<float> &toPts = points3D;

		for (std::size_t i = 1; i < thisMatchIdx.size(); i++) {
			// Assume homogenous with 1 end.
			cv::Vec3f vecFrom(
					fromPts(0, i) - fromPts(0, i - 1),
					fromPts(1, i) - fromPts(1, i - 1),
					fromPts(2, i) - fromPts(2, i - 1)
					);
			cv::Vec3f vecTo(
					toPts(0, i) - toPts(0, i - 1),
					toPts(1, i) - toPts(1, i - 1),
					toPts(2, i) - toPts(2, i - 1)
					);
			scales.push_back(norm(vecTo) / norm(vecFrom));
		}

		// Find the median scale
		std::sort(scales.begin(), scales.end());

		scale = matchFrom.scale * scales[scales.size() / 2];

		poseDiff.t *= scale;
	}
}

ona::Frame::Frame(int id, const cv::Mat &image, const cv::Mat &cameraMatrix, const std::vector<float> &distCoeffs): id_(id), image_(image), cameraMatrix_(cameraMatrix), distCoeffs_(distCoeffs) {
}

bool ona::Frame::compute(cv::FeatureDetector &detector, cv::DescriptorExtractor &extractor) {
	CV_Assert(!image_.empty());

	detector.detect(image_, keyPoints_);
	extractor.compute(image_, keyPoints_, descriptors_);

	return true;
}

int ona::Frame::getId() const {
	return id_;
}

cv::Mat ona::Frame::getImage() const {
	return image_;
}

void ona::Frame::match(SPtr queryFrame, SPtr trainFrame, cv::DescriptorMatcher &matcher, float maxDistance) {
	MatchSPtr onaMatch(new OnaMatch);
	queryFrame->frameMatchesFrom_[trainFrame->id_] = onaMatch;
	trainFrame->frameMatchesTo_[queryFrame->id_] = onaMatch;

	onaMatch->queryFrame = queryFrame;
	onaMatch->trainFrame = trainFrame;

	std::vector<cv::DMatch> matches;
	if (!queryFrame->descriptors_.empty() && !trainFrame->descriptors_.empty()) {
		matcher.match(queryFrame->descriptors_, trainFrame->descriptors_, matches);

		for (const cv::DMatch &match : matches) {
			if (maxDistance == 0 || match.distance <= maxDistance) {
				onaMatch->queryPoints.push_back(queryFrame->keyPoints_[match.queryIdx].pt);
				onaMatch->trainPoints.push_back(trainFrame->keyPoints_[match.trainIdx].pt);
				onaMatch->queryIdx.push_back(match.queryIdx);
				onaMatch->trainIdx.push_back(match.trainIdx);
			}
		}

		if (onaMatch->queryPoints.size() > 0) {
			// undistort and put in normal coordinates
			undistortPoints(onaMatch->queryPoints, onaMatch->queryNormalisedPoints, queryFrame->cameraMatrix_, queryFrame->distCoeffs_);
			undistortPoints(onaMatch->trainPoints, onaMatch->trainNormalisedPoints, trainFrame->cameraMatrix_, trainFrame->distCoeffs_);
		}
	}
}

cv::Mat ona::Frame::findEssentialMatRansac(int id, double ransacMaxDistance, double ransacConfidence) {
	cv::Mat E;

	if (MatchSPtr matchPtr = getMatchById(id)) {
		setEssentialMatRansac(*matchPtr, ransacMaxDistance, ransacConfidence);
		matchPtr->essential.copyTo(E);
	}

	return E;
}

void ona::Frame::findScaleFrom(SPtr commonFrame, int idFrom) {
	if (MatchSPtr matchToThisPtr = getMatchById(commonFrame->id_)) {
		if (MatchSPtr matchToCommonPtr = commonFrame->getMatchById(idFrom)) {
			matchToCommonPtr->setScaleFrom(*matchToThisPtr);
		}
	}
}

ona::Frame::Pose ona::Frame::findPoseDiff(int id) {
	Pose poseDiff;

	if (MatchSPtr matchPtr = getMatchById(id)) {
		setPoseDiff(*matchPtr);
		poseDiff =  matchPtr->poseDiff;
	}

	return poseDiff;
}

cv::Mat ona::Frame::drawMatchedFlowFrom(int id) {
	cv::Mat image;

	if (MatchSPtr matchPtr = getMatchById(id)) {
		drawMatchedFlow(image_, image, matchPtr->trainPoints, matchPtr->queryPoints, matchPtr->inliers);
	}

	return image;
}

ona::Frame::MatchSPtr ona::Frame::getMatchById(int id) {
	IdMatchMapFrom::iterator frameMatchIt = frameMatchesFrom_.find(id);

	if (frameMatchIt != frameMatchesFrom_.end()) {
		return frameMatchIt->second;
#ifndef NDEBUG
	} else {
		std::cout << "Match to frame with id: " << id << " not found" << std::endl;
#endif
	}

	return nullptr;
}

void ona::Frame::setEssentialMatRansac(OnaMatch &match, double ransacMaxDistance, double ransacConfidence) {
	if (match.queryNormalisedPoints.size() > 0) {
		match.essential = findFundamentalMat(match.trainNormalisedPoints, match.queryNormalisedPoints, cv::FM_RANSAC, ransacMaxDistance, ransacConfidence, match.inliers);
	}
}

void ona::Frame::setPoseDiff(OnaMatch &match) {
	if (match.queryNormalisedPoints.size() > 0) {
#ifndef NDEBUG
		std::cout << "Nubmer of good triangulation points: " <<
#endif
		recoverPose(match.essential, match.trainNormalisedPoints, match.queryNormalisedPoints, match.poseDiff.R, match.poseDiff.t, match.points3D, match.inliers)
#ifndef NDEBUG
		<< std::endl
#endif
			;
	}
}
