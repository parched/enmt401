/* Optical navigation aid by James Duley <jagduley@gmail.com> */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "matches.hpp"
#include "pose.hpp"

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
	
	const float maxKeyPointDistance = 0.5;
	const double ransacMaxDistance = 1.;
	const double ransacConfidence = 0.99;

	float tz40KData[] = {520., 0., 320., 0., 520., 240., 0., 0., 1.};
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

	VideoCapture cap;
	Mat lastFrame, currentFrame;
	input_t input;

	int outFps = 10;

	std::string inputFile = arguments.inputFile;
	int cameraIndex;

	std::stringstream inputConvert(inputFile);
	if (inputConvert >> cameraIndex) {
		input = CAM;
		cap.open(cameraIndex);
	} else {
		input = VID;
		cap.open(inputFile);
	}

	if (cap.isOpened()) {
		std::cout << "stream open" << std::endl;

		outFps = cap.get(CV_CAP_PROP_FPS);

		if(!cap.read(currentFrame)) return 1;
	} else {
		input = PIC;
		std::cout << "stream not open" << std::endl;

		lastFrame = imread("/usr/share/opencv/samples/cpp/tsukuba_l.png", CV_LOAD_IMAGE_GRAYSCALE);
		currentFrame = imread("/usr/share/opencv/samples/cpp/tsukuba_r.png", CV_LOAD_IMAGE_GRAYSCALE);
	}

	VideoWriter out;
	if (!arguments.outputFile.empty()) {
		out.open(arguments.outputFile, CV_FOURCC('H', '2', '6', '4'), outFps, currentFrame.size());

		if (out.isOpened()) {
			std::cout << "writing to " << arguments.outputFile << std::endl;
		} else {
			std::cout << "!!! Output video could not be opened" << std::endl;
			return 1;
		}
	}

	Mat img_matches;

	Mat totalR = Mat::eye(3, 3, CV_64F);

	SurfFeatureDetector detector(400);
	vector<KeyPoint> keyPointsLast, keyPointsCurrent;

	SurfDescriptorExtractor extractor;
	Mat descriptorsLast, descriptorsCurrent;
	
	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;

	detector.detect(currentFrame, keyPointsCurrent);
	extractor.compute(currentFrame, keyPointsCurrent, descriptorsCurrent);

	// the camera matrix and distortion coeffs
	Mat K(3, 3, CV_32F, tz40KData);
	vector<float> distCoeffs;

#ifndef NDEBUG
	int frameCounter = 0;
#endif
	//clock_t tFrameAcquired;

	while(waitKey(100) < 0){
		if (input == PIC) {
			Mat lastFrameAlso = lastFrame;
			lastFrame = currentFrame;
			currentFrame = lastFrameAlso;
		} else if (input == VID || input == CAM) {
			lastFrame = currentFrame;
			if (!cap.read(currentFrame)) {
				std::cout << "end of stream" << std::endl;
				break;
			}
		}

		//tFrameAcquired = clock();
#ifndef NDEBUG
		frameCounter++;
		std::cout << "frame " << frameCounter << std::endl;
#endif

		// detecting keypoints
		keyPointsLast = keyPointsCurrent;
		detector.detect(currentFrame, keyPointsCurrent);

		// computing descriptors
		descriptorsLast = descriptorsCurrent;
		extractor.compute(currentFrame, keyPointsCurrent, descriptorsCurrent);

		// matching descriptors
		matcher.match(descriptorsLast, descriptorsCurrent, matches);

		// getting the matched points
		vector<Point2f> lastPoints, currentPoints;
		getMatchedPoints(lastPoints, currentPoints, keyPointsLast, keyPointsCurrent, matches, maxKeyPointDistance);

		// undistort and put in normal coordinates
		vector<Point2f> lastPointsNormal, currentPointsNormal;
		undistortPoints(lastPoints, lastPointsNormal, K, distCoeffs);
		undistortPoints(currentPoints, currentPointsNormal, K, distCoeffs);

		// find the essential matrix E
		vector<uchar> inliers(lastPoints.size());
		Mat E = findFundamentalMat(lastPointsNormal, currentPointsNormal, FM_RANSAC, ransacMaxDistance, ransacConfidence, inliers);

		// get the rotation
		Mat R, t, mask;
		recoverPose(E, lastPointsNormal, currentPointsNormal, R, t, mask);

		// add to tally
		totalR = totalR * R;
#ifndef NDEBUG
		Vec3d eulerAngles;
		getEulerAngles(totalR, eulerAngles);
		std::stringstream poseInfo;
		poseInfo.setf(std::ios::fixed, std::ios::floatfield);
		poseInfo.precision(2);
		poseInfo << "Angle: " << std::setw(6) << eulerAngles(0) << std::setw(6) << eulerAngles(1) << std::setw(6) << eulerAngles(2);
		poseInfo << "   Direction: " << std::setw(6) << t.at<double>(0) << std::setw(6) << t.at<double>(1) << std::setw(6) << t.at<double>(2);
		std::cout << poseInfo.str() << std::endl;
#endif
		// drawing the results
		namedWindow("matches", 1);
		drawMatchedFlow(currentFrame, img_matches, lastPoints, currentPoints, inliers);
#ifndef NDEBUG
		putText(img_matches, poseInfo.str(), Point(15, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0xf7, 0x2e, 0xfe));
#endif

		imshow("matches", img_matches);

		out.write(img_matches);
	}

	return 0;
}
