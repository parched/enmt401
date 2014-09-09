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

//#include <iostream>

#include "pose.hpp"

ona::Frame::Frame(int id, const cv::Mat &image, const cv::Mat &cameraMatrix, const std::vector<float> &distCoeffs): id_(id), image_(image), cameraMatrix_(cameraMatrix), distCoeffs_(distCoeffs) {
}

void ona::Frame::compute(const cv::FeatureDetector &detector, const cv::DescriptorExtractor &extractor) {
	CV_Assert(!image_.empty());

	detector.detect(image_, keyPoints_);
	extractor.compute(image_, keyPoints_, descriptors_);
}

void ona::Frame::compute(const cv::Feature2D &detectorAndExtractor) {
	CV_Assert(!image_.empty());

	detectorAndExtractor(image_, cv::noArray(), keyPoints_, descriptors_);
}

int ona::Frame::getId() const {
	return id_;
}

cv::Mat ona::Frame::getImage() const {
	return image_;
}
