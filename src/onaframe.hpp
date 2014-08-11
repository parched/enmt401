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
		typedef std::unique_ptr<OnaFrame> UPtr;
		typedef std::shared_ptr<OnaFrame> SPtr;
		typedef std::weak_ptr<OnaFrame> WPtr;

		/**
		 * \brief OnaFrame constructor.
		 *
		 * \param id The id of the frame which should be unique.
		 * \param cameraMatrix The corresponding camera matirx.
		 * \param distCoeffs The corresponding radial distortion coefficients.
		 */
		OnaFrame(int id, const cv::Mat &image, const cv::Mat &cameraMatrix, const std::vector<float> &distCoeffs);

		/**
		 * \brief Gets the id of the OnaFrame
		 *
		 * \return The id.
		 */
		int getId() const;

		/**
		 * \brief Gets the image of the OnaFrame.
		 *
		 * \return The image.
		 */
		cv::Mat getImage() const;

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
		bool match(WPtr frameToMatchTo, cv::DescriptorMatcher &matcher, float maxDistance);


	protected:
		int _id;
		cv::Mat _image;
		cv::Mat _cameraMatrix;
		std::vector<float> _distCoeffs;
		std::vector<cv::KeyPoint> _keyPoints;
		cv::Mat _descriptors;

		struct OnaMatch {
			public:
				WPtr trainFrame;
				std::vector<int> queryIdx, trainIdx;
				std::vector<cv::Point2f> queryNormalisedPoints, trainNormalisedPoints;
		};

		typedef std::map<int, OnaMatch> IdMatchMap;
		IdMatchMap frameMatches;

};
#endif
