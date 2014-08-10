
#include "onaframe.hpp"

using namespace cv;

bool OnaFrame::compute(FeatureDetector &detector, DescriptorExtractor &extractor) {
	CV_Assert(!image.empty());

	detector.detect(image, keyPoints);
	extractor.compute(image, keyPoints, descriptors);

	return true;
}
