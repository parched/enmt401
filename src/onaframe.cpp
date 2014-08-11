
#include "onaframe.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

OnaFrame::OnaFrame(int id, Mat cameraMatrix, std::vector<float> distCoeffs): cameraMatrix(cameraMatrix), distCoeffs(distCoeffs), _id(id) {
}

bool OnaFrame::compute(FeatureDetector &detector, DescriptorExtractor &extractor) {
	CV_Assert(!image.empty());

	detector.detect(image, keyPoints);
	extractor.compute(image, keyPoints, descriptors);

	return true;
}

int OnaFrame::getId() const {
	return _id;
}

bool OnaFrame::match(std::weak_ptr<OnaFrame> frameToMatchTo, DescriptorMatcher &matcher, float maxDistance) {
	if (std::shared_ptr<OnaFrame> trainFrame = frameToMatchTo.lock()) {
		OnaMatch onaMatch;
		onaMatch.trainFrame = frameToMatchTo;

		std::vector<DMatch> matches;
		matcher.match(descriptors, trainFrame->descriptors, matches);

		vector<Point2f> queryPoints, trainPoints;
		for (const DMatch &match : matches) {
			if (maxDistance == 0 || match.distance <= maxDistance) {
				queryPoints.push_back(keyPoints[match.queryIdx].pt);
				trainPoints.push_back(trainFrame->keyPoints[match.trainIdx].pt);
				onaMatch.queryIdx.push_back(match.queryIdx);
				onaMatch.trainIdx.push_back(match.trainIdx);
			}
		}

		// undistort and put in normal coordinates
		undistortPoints(queryPoints, onaMatch.queryNormalisedPoints, cameraMatrix, distCoeffs);
		undistortPoints(trainPoints, onaMatch.trainNormalisedPoints, trainFrame->cameraMatrix, trainFrame->distCoeffs);

		frameMatches.push_back(onaMatch);
	} else {
		return false;
	}

	return true;
}
