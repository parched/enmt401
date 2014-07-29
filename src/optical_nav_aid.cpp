/* Optical navigation aid by James Duley <jagduley@gmail.com> */

#include <iostream>
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
		{"input", 'i', "FILE", 0, "Input file", 0},
		{0, 0, 0, 0, 0, 0}
	};
	
	const float maxKeyPointDistance = 0.5;
	const double ransacMaxDistance = 1.;
	const double ransacConfidence = 0.99;

	float tz40KData[] = {520., 0., 320., 0., 520., 240., 0., 0., 1.};
}

struct arguments {
	std::string inputFile;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
	struct arguments *arguments = (struct arguments *)state->input;

	switch (key) {
		case 'i':
			arguments->inputFile = arg;
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
	arguments.inputFile = "-";

	argp_parse (&argp, argc, argv, 0, 0, &arguments);

	VideoCapture cap;
	Mat lastFrame, currentFrame;
	input_t input;

	cap.open(arguments.inputFile);
	if (!cap.isOpened()) {
		input = PIC;
		std::cout << "video not open" << std::endl;
		lastFrame = imread("/usr/share/opencv/samples/cpp/tsukuba_l.png", CV_LOAD_IMAGE_GRAYSCALE);
		currentFrame = imread("/usr/share/opencv/samples/cpp/tsukuba_r.png", CV_LOAD_IMAGE_GRAYSCALE);
	} else {
		input = VID;
		std::cout << "video open" << std::endl;
		if(!cap.read(currentFrame)) return 1;
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
		} else if (input == VID) {
			lastFrame = currentFrame;
			if (!cap.read(currentFrame)) break;
		}

		//tFrameAcquired = clock();
#ifndef NDEBUG
		std::cout << "frame " << frameCounter << std::endl;
		frameCounter++;
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
#ifndef NDEBUG
		if (checkCoherentRotation(R)) {
			std::cout << R << std::endl;
		}
#endif

		// add to tally
		totalR = totalR * R;
#ifndef NDEBUG
		printEulerAngles(totalR);
#endif
		// drawing the results
		namedWindow("matches", 1);
		drawMatchedFlow(currentFrame, img_matches, lastPoints, currentPoints, inliers);
		imshow("matches", img_matches);
	}

	return 0;
}
