
#include "onaframe.hpp"

using namespace cv;

OnaFrame::OnaFrame(Mat cameraMatrix, std::vector<float> distCoeffs): cameraMatrix(cameraMatrix), distCoeffs(distCoeffs) {
}

bool OnaFrame::compute(FeatureDetector &detector, DescriptorExtractor &extractor) {
	CV_Assert(!image.empty());

	detector.detect(image, keyPoints);
	extractor.compute(image, keyPoints, descriptors);

	return true;
}
