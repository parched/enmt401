/* Optical navigation aid by James Duley <jagduley@gmail.com> */

#include <stdlib.h>
#include <stdio.h>

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

	Mat img1 = imread("/usr/share/opencv/samples/cpp/tsukuba_l.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("/usr/share/opencv/samples/cpp/tsukuba_r.png", CV_LOAD_IMAGE_GRAYSCALE);
	if(img1.empty() || img2.empty())
	{
	    printf("Can't read one of the images\n");
	    return -1;
	}

	// detecting keypoints
	SurfFeatureDetector detector(3300);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(img1, keypoints1);
	detector.detect(img2, keypoints2);

	// computing descriptors
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keypoints1, descriptors1);
	extractor.compute(img2, keypoints2, descriptors2);

	// matching descriptors
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// drawing the results
	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
	imshow("matches", img_matches);
	waitKey(0);

	return 0;
}
