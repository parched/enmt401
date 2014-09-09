/**
 * \file optical_nav_aid.cpp
 * \brief An optical navigation aid.
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

#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "matches.hpp"
#include "pose.hpp"
#include "onaframe.hpp"
#include "onamatch.hpp"

/* This needs to go last or it causes an error with c++11 */
#include <argp.h>

const char *argp_program_version = "Optical Navigation Aid v?";
const char *argp_program_bug_address = "<jagduley@gmail.com>";

namespace {
	/* Program documentation. */
	const char doc[] = "An optical navigation aid for a UAV.";

	const struct argp_option options[] = {
		{"input", 'i', "FILE", 0, "Input file or camera index", 0},
		{"output", 'o', "FILE", 0, "Output file", 0},
		{0, 0, 0, 0, 0, 0}
	};
	
	const float maxDescriptorDistance = 0.0;
	const double ransacMaxDistance = 1.;
	const double ransacConfidence = 0.99;

	float tz40KData[] = {520., 0., 320., 0., 520., 240., 0., 0., 1.};
	float fireflyData[] = {1.3840577621390364e+03, 0., 3.0934395664809961e+02, 0., 1.3834780739724592e+03, 2.4440820692178860e+02, 0., 0., 1.};
	std::vector<float> fireflyDistCoeffs { -2.8638478534592271e-01, -7.1981969625655881e-02, -3.2364565871468563e-03, 1.5622879005666110e-03, 4.9534384707813928e+00};
}

struct arguments {
	std::string inputFile;
	std::string outputFile;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
	struct arguments *arguments = (struct arguments *)state->input;

	switch (key) {
		case 'i':
			arguments->inputFile = arg;
			break;
		case 'o':
			arguments->outputFile = arg;
			break;
		default:
			return ARGP_ERR_UNKNOWN;
	}

	return 0;
}

static struct argp argp = {options, parse_opt, 0, doc, 0, 0, 0};

enum input_t {PIC, VID, CAM};

int main(int argc, char **argv) {

	struct arguments arguments;

	argp_parse (&argp, argc, argv, 0, 0, &arguments);

	// the camera matrix and distortion coeffs
	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	std::vector<float> distCoeffs;

	int frameCounter = 0;

	ona::Frame::UPtr currentFrame;
	ona::Frame::UPtr lastFrame;

	cv::VideoCapture cap;
	input_t input;

	int outFps = 10;

	std::string inputFile = arguments.inputFile;
	int cameraIndex;

	std::stringstream inputConvert(inputFile);
	if (inputConvert >> cameraIndex) {
		input = CAM;
		std::cout << "Input from camera " << arguments.inputFile << std::endl;
		cap.open(cameraIndex);
		K = cv::Mat(3, 3, CV_32F, tz40KData);
	} else {
		input = VID;
		std::cout << "Input from file " << arguments.inputFile << std::endl;
		cap.open(inputFile);
		K = cv::Mat(3, 3, CV_32F, fireflyData);
		distCoeffs = fireflyDistCoeffs;
	}

	if (cap.isOpened()) {
		std::cout << "stream open" << std::endl;

		outFps = cap.get(cv::CAP_PROP_FPS);

		cv::Mat image;
		if(cap.read(image)) {
			currentFrame = ona::Frame::UPtr(new ona::Frame(frameCounter, image, K, distCoeffs));
		} else {
			std::cerr << "Could not read an image from capture" << std::endl;
			return 1;
		}
	} else {
		input = PIC;
		std::cout << "stream not open" << std::endl;

		lastFrame = ona::Frame::UPtr(new ona::Frame(frameCounter - 1, cv::imread("/usr/share/OpenCV/samples/cpp/tsukuba_l.png", cv::IMREAD_GRAYSCALE), K, distCoeffs));
		currentFrame = ona::Frame::UPtr(new ona::Frame(frameCounter, cv::imread("/usr/share/OpenCV/samples/cpp/tsukuba_r.png", cv::IMREAD_GRAYSCALE), K, distCoeffs));
	}

	cv::VideoWriter out;
	if (!arguments.outputFile.empty()) {
		out.open(arguments.outputFile, cv::VideoWriter::fourcc('H', '2', '6', '4'), outFps, currentFrame->getImage().size());

		if (out.isOpened()) {
			std::cout << "writing to " << arguments.outputFile << std::endl;
		} else {
			std::cout << "!!! Output video could not be opened" << std::endl;
			return 1;
		}
	} else {
		std::cout << "Not writing output." << std::endl;
	}

	cv::Mat totalR = cv::Mat::eye(3, 3, CV_64F);

	cv::BRISK detector;

	cv::BRISK &extractor = detector;

	cv::BFMatcher matcher(cv::NORM_L2, true);

	currentFrame->compute(detector, extractor);

	//clock_t tFrameAcquired;
	
	ona::Match::UPtr lastMatch, currentMatch;

#ifndef NDEBUG
	std::cout << "Starting main loop." << std::endl;
#endif
	while(cv::waitKey(100) < 0){
		frameCounter++;
#ifndef NDEBUG
		std::cout << "frame " << frameCounter << std::endl;
#endif
		cv::Mat newImage;

		if (input == PIC) {
			newImage = lastFrame->getImage();
		} else if (input == VID || input == CAM) {
			if (!cap.read(newImage)) {
				std::cout << "end of stream" << std::endl;
				break;
			}
		}

		lastFrame = std::move(currentFrame);
		currentFrame = ona::Frame::UPtr(new ona::Frame(frameCounter, newImage, K, distCoeffs));

		//tFrameAcquired = clock();

		// computing descriptors
		currentFrame->compute(detector, extractor);

		// compute the match with the last frame.
		lastMatch = std::move(currentMatch);
		currentMatch = ona::Match::UPtr(new ona::Match(*currentFrame, *lastFrame, matcher, maxDescriptorDistance));

		// find the essential matrix E and pose difference
		currentMatch->compute(ransacMaxDistance, ransacConfidence);

		// wait for 2 matches
		if (lastMatch) {
			// set the scale
			currentMatch->setScaleFrom(*lastMatch);
		}

		// get the rotation
		ona::Match::Pose poseDiff = currentMatch->getPoseDiff();

		if (!poseDiff.R.empty()) {
			// add to tally
			totalR = totalR * poseDiff.R;
#ifndef NDEBUG
			cv::Vec3d eulerAngles;
			getEulerAngles(totalR, eulerAngles);
			std::stringstream poseInfo;
			poseInfo.setf(std::ios::fixed, std::ios::floatfield);
			poseInfo.precision(2);
			poseInfo << "Angle: " << std::setw(8) << eulerAngles(0) << std::setw(8) << eulerAngles(1) << std::setw(8) << eulerAngles(2);
			poseInfo << "   Direction: " 
				<< std::setw(6) << poseDiff.t.at<double>(0) 
				<< std::setw(6) << poseDiff.t.at<double>(1) 
				<< std::setw(6) << poseDiff.t.at<double>(2);
			std::cout << poseInfo.str() << std::endl;
#endif
			// drawing the results
			cv::Mat imgMatches = currentMatch->drawFlow(lastFrame->getImage());
#ifndef NDEBUG
			putText(imgMatches, poseInfo.str(), cv::Point(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0xf7, 0x2e, 0xfe));
#endif

			cv::namedWindow("matches", 1);
			imshow("matches", imgMatches);

			out.write(imgMatches);
		}
	}

	return 0;
}
