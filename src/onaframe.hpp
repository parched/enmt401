#ifndef ONAFRAME_HPP
#define ONAFRAME_HPP

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

/**
 * \brief An optical navgiation aid frame.
 */
class OnaFrame {
	public:
		cv::Mat image;
		cv::Mat cameraMatrix;
		std::vector<float> distCoeffs;
		std::vector<cv::KeyPoint> keyPoints;
		cv::Mat descriptors;

		/**
		 * \brief OnaFrame constructor.
		 *
		 * \param cameraMatrix The corresponding camera matirx.
		 * \param distCoeffs The corresponding radial distortion coefficients.
		 */
		OnaFrame(cv::Mat cameraMatrix, std::vector<float> distCoeffs);

		/**
		 * \brief Compute the keyPoints descriptors.
		 *
		 * \param detector The FeatureDetector to use.
		 * \param extractor The DescriptorExtractor to use.
		 *
		 * \return Successful.
		 */
		bool compute(cv::FeatureDetector &detector, cv::DescriptorExtractor &extractor);

		/**
		 * \brief Match descriptors to another frame.
		 *
		 * \param frameToMatchTo The train frame to match to.
		 * \param matcher The DescriptorMatcher to use.
		 * \param maxDistance The maximum allowable description distance to allow.
		 *
		 * \return Successfulness.
		 */
		bool match(std::weak_ptr<OnaFrame> frameToMatchTo, cv::DescriptorMatcher &matcher, float maxDistance);


	protected:
		struct OnaMatch {
			public:
				std::weak_ptr<OnaFrame> trainFrame;
				std::vector<int> queryIdx, trainIdx;
				std::vector<cv::Point2f> queryNormalisedPoints, trainNormalisedPoints;
		};

		std::vector<OnaMatch> frameMatches;

};
#endif
