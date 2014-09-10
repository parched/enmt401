/**
 * \file onamatch.cpp
 * \brief Optical navigation aid Match class implementation.
 * \author James Duley
 * \version 1.0
 * \date 2014-09-07
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

#include "onamatch.hpp"

#include <iostream>
#include <algorithm>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "onaframe.hpp"
#include "pose.hpp"
#include "matches.hpp"

ona::Match::Match(const Frame &queryFrame, const Frame &trainFrame, cv::DescriptorMatcher &matcher, float maxDistance) {
	queryFrameId_ = queryFrame.id_;
	trainFrameId_ = trainFrame.id_;

	std::vector<cv::DMatch> matches;
	if (!queryFrame.descriptors_.empty() && !trainFrame.descriptors_.empty()) {
		matcher.match(queryFrame.descriptors_, trainFrame.descriptors_, matches);

		for (const cv::DMatch &match : matches) {
			if (maxDistance == 0 || match.distance <= maxDistance) {
				queryPoints_.push_back(queryFrame.keyPoints_[match.queryIdx].pt);
				trainPoints_.push_back(trainFrame.keyPoints_[match.trainIdx].pt);
				queryIdx_.push_back(match.queryIdx);
				trainIdx_.push_back(match.trainIdx);
			}
		}

		if (queryPoints_.size() > 0) {
			// undistort and put in normal coordinates
			cv::undistortPoints(queryPoints_, queryNormalisedPoints_, queryFrame.cameraMatrix_, queryFrame.distCoeffs_);
			cv::undistortPoints(trainPoints_, trainNormalisedPoints_, trainFrame.cameraMatrix_, trainFrame.distCoeffs_);
		}
	}

	// set the default scale to 1.0
	scale_ = 1.0;
}

int ona::Match::compute(double ransacMaxDistance, double ransacConfidence) {
	setEssentialMatRansac(ransacMaxDistance, ransacConfidence);
	if (!essential_.empty()) {
		setPoseDiff();
	}
	return inliers_.size();
}

void ona::Match::setScaleFrom(const Match &matchFrom) {
	// check for a common frame and essentail matrices found
	if (trainFrameId_ == matchFrom.queryFrameId_ && !essential_.empty() && !matchFrom.essential_.empty() && matchFrom.scale_ != 0.0) {
		std::vector<int> thisMatchIdx, fromMatchIdx;

		auto fromStartIter = matchFrom.queryIdx_.begin();
		auto fromEndIter = matchFrom.queryIdx_.end();

		for (std::size_t i = 0; i < inliers_.size(); i++) {
			if (inliers_[i]) {
				auto fromFoundIter = std::find(fromStartIter, fromEndIter, trainIdx_[i]);
				if (fromFoundIter != fromEndIter) {
					auto fromIterIdx = std::distance(matchFrom.queryIdx_.begin(), fromFoundIter);
					if (matchFrom.inliers_[fromIterIdx]) {
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

			const cv::Mat_<float> &fromPts = matchFrom.points3D_;
			const cv::Mat_<float> &toPts = points3D_;

			for (std::size_t i = 1; i < thisMatchIdx.size(); i++) {
				// Assume homogenous with 1 end.
				const cv::Vec3f vecFrom(
						fromPts(0, fromMatchIdx[i]) - fromPts(0, fromMatchIdx[i - 1]),
						fromPts(1, fromMatchIdx[i]) - fromPts(1, fromMatchIdx[i - 1]),
						fromPts(2, fromMatchIdx[i]) - fromPts(2, fromMatchIdx[i - 1])
						);
				const cv::Vec3f vecTo(
						toPts(0, thisMatchIdx[i]) - toPts(0, thisMatchIdx[i - 1]),
						toPts(1, thisMatchIdx[i]) - toPts(1, thisMatchIdx[i - 1]),
						toPts(2, thisMatchIdx[i]) - toPts(2, thisMatchIdx[i - 1])
						);
				scales.push_back(norm(vecTo) / norm(vecFrom));
			}

			// Find the median scale
			std::sort(scales.begin(), scales.end());

			scale_ = matchFrom.scale_ * scales[scales.size() / 2];

			poseDiff_.t *= scale_;
		}
#ifndef NDEBUG
	} else {
		std::cout << "No common Frame between matches || essential matrices not found." << std::endl;
#endif
	}
}

ona::Match::Pose ona::Match::getPoseDiff() {
	return poseDiff_;
}

cv::Mat ona::Match::drawFlow(const cv::Mat &image) {
	cv::Mat drawnImage;

	drawMatchedFlow(image, drawnImage, trainPoints_, queryPoints_, inliers_);

	return drawnImage;
}

void ona::Match::setEssentialMatRansac(double ransacMaxDistance, double ransacConfidence) {
	int numPoints = queryNormalisedPoints_.size();

	if (numPoints > 5) {
		essential_ = findEssentialMat(trainNormalisedPoints_, queryNormalisedPoints_, cv::RANSAC, ransacMaxDistance, ransacConfidence, inliers_);
	} else if (numPoints > 0) {
		essential_ = findFundamentalMat(trainNormalisedPoints_, queryNormalisedPoints_, cv::FM_RANSAC, ransacMaxDistance, ransacConfidence, inliers_);
	}
}

void ona::Match::setPoseDiff() {
	if (queryNormalisedPoints_.size() > 0) {
#ifndef NDEBUG
		std::cout << "Nubmer of good triangulation points: " <<
#endif
		cv::recoverPose(essential_, trainNormalisedPoints_, queryNormalisedPoints_, poseDiff_.R, poseDiff_.t, points3D_, inliers_)
#ifndef NDEBUG
		<< std::endl
#endif
			;
	}
}
