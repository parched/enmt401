#ifndef ONAFRAME_HPP
#define ONAFRAME_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

/**
 * \brief An optical navgiation aid frame.
 */
class OnaFrame {
	public:
		cv::Mat image;
		std::vector<cv::KeyPoint> keyPoints;
		cv::Mat descriptors;

		/**
		 * \brief Compute the keyPoints descriptors.
		 *
		 * \param detector The FeatureDetector to use.
		 * \param extractor The DescriptorExtractor to use.
		 *
		 * \return Successful.
		 */
		bool compute(cv::FeatureDetector &detector, cv::DescriptorExtractor &extractor);


	protected:
};
#endif
