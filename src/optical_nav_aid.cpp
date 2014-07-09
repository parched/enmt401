/* Optical navigation aid by James Duley <jagduley@gmail.com> */

#include <iostream>
#include <ctime>

#include <argp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;

const char *argp_program_version = "Optical Navigation Aid v?";
const char *argp_program_bug_address = "<jagduley@gmail.com>";

/* Program documentation. */
static char doc[] = 
"An optical navigation aid for a UAV.";


static struct argp argp = {0, 0, 0, doc};

int main(int argc, char **argv) {
	argp_parse (&argp, argc, argv, 0, 0, 0);

	Mat lastFrame = imread("/usr/share/opencv/samples/cpp/tsukuba_l.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat currentFrame = imread("/usr/share/opencv/samples/cpp/tsukuba_r.png", CV_LOAD_IMAGE_GRAYSCALE);
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
		Mat lastFrameAlso = lastFrame;
		lastFrame = currentFrame;

		currentFrame = lastFrameAlso;

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
		drawMatches(lastFrame, keyPointsLast, currentFrame, keyPointsCurrent, matches, img_matches);
		imshow("matches", img_matches);
	}


	return 0;
}
