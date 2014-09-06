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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "matches.hpp"
#include "pose.hpp"
#include "onaframe.hpp"

/* This needs to go last or it causes an error with c++11 */
#include <argp.h>

using namespace cv;

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
	
	const float maxDescriptorDistance = 0.5;
	const double ransacMaxDistance = 1.;
	const double ransacConfidence = 0.99;

	float tz40KData[] = {520., 0., 320., 0., 520., 240., 0., 0., 1.};
	float fireflyData[] = {1.3840577621390364e+03, 0., 3.0934395664809961e+02, 0., 1.3834780739724592e+03, 2.4440820692178860e+02, 0., 0., 1.};
	vector<float> fireflyDistCoeffs { -2.8638478534592271e-01, -7.1981969625655881e-02, -3.2364565871468563e-03, 1.5622879005666110e-03, 4.9534384707813928e+00};
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
	Mat K = Mat::eye(3, 3, CV_32F);
	vector<float> distCoeffs;

	int frameCounter = 0;

	OnaFrame::SPtr currentFrame;
	OnaFrame::SPtr lastFrame;

	VideoCapture cap;
	input_t input;

	int outFps = 10;

	std::string inputFile = arguments.inputFile;
	int cameraIndex;

	std::stringstream inputConvert(inputFile);
	if (inputConvert >> cameraIndex) {
		input = CAM;
		std::cout << "Input from camera " << arguments.inputFile << std::endl;
		cap.open(cameraIndex);
		K = Mat(3, 3, CV_32F, tz40KData);
	} else {
		input = VID;
		std::cout << "Input from file " << arguments.inputFile << std::endl;
		cap.open(inputFile);
		K = Mat(3, 3, CV_32F, fireflyData);
		distCoeffs = fireflyDistCoeffs;
	}

	if (cap.isOpened()) {
		std::cout << "stream open" << std::endl;

		outFps = cap.get(CV_CAP_PROP_FPS);

		Mat image;
		if(cap.read(image)) {
			currentFrame = OnaFrame::SPtr(new OnaFrame(frameCounter, image, K, distCoeffs));
		} else {
			std::cerr << "Could not read an image from capture" << std::endl;
			return 1;
		}
	} else {
		input = PIC;
		std::cout << "stream not open" << std::endl;

		lastFrame = OnaFrame::SPtr(new OnaFrame(frameCounter - 1, imread("/usr/share/opencv/samples/cpp/tsukuba_l.png", CV_LOAD_IMAGE_GRAYSCALE), K, distCoeffs));
		currentFrame = OnaFrame::SPtr(new OnaFrame(frameCounter, imread("/usr/share/opencv/samples/cpp/tsukuba_r.png", CV_LOAD_IMAGE_GRAYSCALE), K, distCoeffs));
	}

	VideoWriter out;
	if (!arguments.outputFile.empty()) {
		out.open(arguments.outputFile, CV_FOURCC('H', '2', '6', '4'), outFps, currentFrame->getImage().size());

		if (out.isOpened()) {
			std::cout << "writing to " << arguments.outputFile << std::endl;
		} else {
			std::cout << "!!! Output video could not be opened" << std::endl;
			return 1;
		}
	} else {
		std::cout << "Not writing output." << std::endl;
	}

	Mat totalR = Mat::eye(3, 3, CV_64F);

	SurfFeatureDetector detector(400);

	SurfDescriptorExtractor extractor;
	
	BFMatcher matcher(NORM_L2, true);

	currentFrame->compute(detector, extractor);

	//clock_t tFrameAcquired;

#ifndef NDEBUG
	std::cout << "Starting main loop." << std::endl;
#endif
	while(waitKey(100) < 0){
		frameCounter++;
#ifndef NDEBUG
		std::cout << "frame " << frameCounter << std::endl;
#endif
		Mat newImage;

		if (input == PIC) {
			newImage = lastFrame->getImage();
		} else if (input == VID || input == CAM) {
			if (!cap.read(newImage)) {
				std::cout << "end of stream" << std::endl;
				break;
			}
		}

		lastFrame = currentFrame;
		currentFrame = OnaFrame::SPtr(new OnaFrame(frameCounter, newImage, K, distCoeffs));

		//tFrameAcquired = clock();

		// computing descriptors
		currentFrame->compute(detector, extractor);

		// matching descriptors
		OnaFrame::match(currentFrame, lastFrame, matcher, maxDescriptorDistance);

		// find the essential matrix E
		Mat E = currentFrame->findEssentialMatRansac(lastFrame->getId(), ransacMaxDistance, ransacConfidence);

		// get the rotation
		OnaFrame::Pose poseDiff = currentFrame->findPoseDiff(lastFrame->getId());

		// set the scale
		currentFrame->findScaleFrom(lastFrame, lastFrame->getId() - 1);

		if (!poseDiff.R.empty()) {
			// add to tally
			totalR = totalR * poseDiff.R;
#ifndef NDEBUG
			Vec3d eulerAngles;
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
			Mat imgMatches = currentFrame->drawMatchedFlowFrom(lastFrame->getId());
#ifndef NDEBUG
			putText(imgMatches, poseInfo.str(), Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0xf7, 0x2e, 0xfe));
#endif

			namedWindow("matches", 1);
			imshow("matches", imgMatches);

			out.write(imgMatches);
		}
	}

	return 0;
}
