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

		struct Pose {
			public:
				cv::Mat R;
				cv::Mat t;
		};

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
		 * \param frameToMatchFrom The query frame to match to.
		 * \param frameToMatchTo The train frame to match to.
		 * \param matcher The DescriptorMatcher to use.
		 * \param maxDistance The maximum allowable description distance to allow.
		 */
		static void match(SPtr frameToMatchFrom, SPtr frameToMatchTo, cv::DescriptorMatcher &matcher, float maxDistance);

		/**
		 * \brief Find the essential matrix from frame by id to this one.
		 *
		 * \param id The id of the frame to calculate from.
		 * \param ransacMaxDistance The maximum permissible distance is pixels to epipolar line.
		 * \param ransacConfidence The desired probability that the answer is correct.
		 *
		 * \return The essentail matrix.
		 */
		cv::Mat findEssentialMatRansac(int id, double ransacMaxDistance, double ransacConfidence);

		/**
		 * \brief Finds the pose difference w.r.t a frame.
		 *
		 * \param id The id of the frame to find the pose difference w.r.t.
		 *
		 * \return Pose transformation.
		 */
		Pose findPoseDiff(int id);

		/**
		 * \brief Draws the matched flow from a frame.
		 *
		 * \param id The id of the frame to draw the flow from.
		 *
		 * \return The image with the flow on it.
		 */
		cv::Mat drawMatchedFlowFrom(int id);

	protected:
		struct OnaMatch {
			public:
				WPtr queryFrame;
				WPtr trainFrame;
				std::vector<int> queryIdx, trainIdx;
				std::vector<cv::Point2f> queryPoints, trainPoints;
				std::vector<cv::Point2f> queryNormalisedPoints, trainNormalisedPoints;
				cv::Mat essential;
				std::vector<uchar> inliers;
				Pose poseDiff;
		};

		typedef std::map<int, OnaMatch> IdMatchMap;

		int _id;
		cv::Mat _image;
		cv::Mat _cameraMatrix;
		std::vector<float> _distCoeffs;
		std::vector<cv::KeyPoint> _keyPoints;
		cv::Mat _descriptors;
		IdMatchMap frameMatches;

		/**
		 * \brief Gets the OnaMatch from the id.
		 *
		 * \param id The id.
		 *
		 * \return A pointer to the match, null_ptr if none.
		 */
		OnaMatch *getMatchById(int id);

		/**
		 * \brief Sets the essential matrix from frame by id to this one.
		 *
		 * \param match The match to calculate and set in.
		 * \param ransacMaxDistance The maximum permissible distance is pixels to epipolar line.
		 * \param ransacConfidence The desired probability that the answer is correct.
		 */
		void setEssentialMatRansac(OnaMatch &match, double ransacMaxDistance, double ransacConfidence);

		/**
		 * \brief Sets the pose difference for a match.
		 *
		 * \param match The match to calculate the pose difference for.
		 */
		void setPoseDiff(OnaMatch &match);
};
#endif
