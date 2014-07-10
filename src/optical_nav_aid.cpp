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

/* Program documentation. */
static char doc[] = 
"An optical navigation aid for a UAV.";

static struct argp_option options[] = {
	{"input", 'i', "FILE", 0, "Input file"},
	{0}
};

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

void drawMatchedFlow(const Mat &currentFrame, const vector<KeyPoint> &keyPointsLast, const vector<KeyPoint> &keyPointsCurrent, const vector<DMatch> &matches, Mat &imgMatches) {

	currentFrame.copyTo(imgMatches);

	for (const DMatch &match : matches) {
		Point2f lastPoint = keyPointsLast[match.queryIdx].pt;
		Point2f currentPoint = keyPointsCurrent[match.trainIdx].pt;

		line(imgMatches, lastPoint, currentPoint, Scalar(0, 0, 255));
		circle(imgMatches, currentPoint, 5, Scalar(0, 0, 255));
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

	SurfFeatureDetector detector(3300);
	vector<KeyPoint> keyPointsLast, keyPointsCurrent;

	SurfDescriptorExtractor extractor;
	Mat descriptorsLast, descriptorsCurrent;
	
	BFMatcher matcher(NORM_L2);
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

		// drawing the results
		namedWindow("matches", 1);
		drawMatchedFlow(currentFrame, keyPointsLast, keyPointsCurrent, matches, img_matches);
		imshow("matches", img_matches);
	}

	return 0;
}
