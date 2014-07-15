/* Optical navigation aid by James Duley <jagduley@gmail.com> */

#include <iostream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

/* This needs to go last or it causes an error with c++11 */
#include <argp.h>

using namespace cv;

const char *argp_program_version = "Optical Navigation Aid v?";
const char *argp_program_bug_address = "<jagduley@gmail.com>";

namespace {
	/* Program documentation. */
	const char doc[] = "An optical navigation aid for a UAV.";

	const struct argp_option options[] = {
		{"input", 'i', "FILE", 0, "Input file"},
		{0}
	};
	
	const float maxKeyPointDistance = 0.5;
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

static struct argp argp = {options, parse_opt, 0, doc};

void getMatchedPoints(vector<Point2f> &lastPoints, vector<Point2f> &currentPoints, const vector<KeyPoint> &keyPointsLast, const vector<KeyPoint> &keyPointsCurrent, const vector<DMatch> &matches, float maxDistance) {

	for (const DMatch &match : matches) {
		if (maxDistance == 0 || match.distance <= maxDistance) {
			lastPoints.push_back(keyPointsLast[match.queryIdx].pt);
			currentPoints.push_back(keyPointsCurrent[match.trainIdx].pt);
		}
	}
}

void drawMatchedFlow(const Mat &currentFrame, const vector<Point2f> &lastPoints, const vector<Point2f> &currentPoints, Mat &imgMatches) {
	currentFrame.copyTo(imgMatches);

	assert(lastPoints.size() == currentPoints.size());

	for (vector<Point2f>::size_type i = 0; i != lastPoints.size(); i++) {
		line(imgMatches, lastPoints[i], currentPoints[i], Scalar(0, 0, 255));
		circle(imgMatches, currentPoints[i], 2, Scalar(0, 0, 255));
	}
}

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
	}
	else {
		input = VID;
		std::cout << "video open" << std::endl;
		if(!cap.read(currentFrame)) return 1;
	}

	Mat img_matches;

	SurfFeatureDetector detector(400);
	vector<KeyPoint> keyPointsLast, keyPointsCurrent;

	SurfDescriptorExtractor extractor;
	Mat descriptorsLast, descriptorsCurrent;
	
	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;

	detector.detect(currentFrame, keyPointsCurrent);
	extractor.compute(currentFrame, keyPointsCurrent, descriptorsCurrent);
#ifndef NDEBUG
	int frameCounter = 0;
#endif
	//clock_t tFrameAcquired;

	while(waitKey(100) < 0){
		if (input == PIC) {
			Mat lastFrameAlso = lastFrame;
			lastFrame = currentFrame;
			currentFrame = lastFrameAlso;
		}
		else if (input == VID) {
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

		// drawing the results
		namedWindow("matches", 1);
		drawMatchedFlow(currentFrame, lastPoints, currentPoints, img_matches);
		imshow("matches", img_matches);
	}

	return 0;
}
