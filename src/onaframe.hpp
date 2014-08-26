/**
 * \file onaframe.hpp
 * \brief An optical navigation aid frame class header.
 * \author James Duley
 * \version 1.0
 * \date 2014-08-26
 */

/* Copyright (C) 
 * 2014 - James Duley
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 * 
 */

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
		typedef std::unique_ptr<OnaFrame> UPtr; /**< A unique pointer to an OnaFrame */
		typedef std::shared_ptr<OnaFrame> SPtr; /**< A shared pointer to an OnaFrame */
		typedef std::weak_ptr<OnaFrame> WPtr; /**< A weak pointer to an OnaFrame */

		/**
		 * \brief A pose transformation.
		 */
		struct Pose {
			cv::Mat R; /**< Rotation. */
			cv::Mat t; /**< Translation. */
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
		/**
		 * \brief A match between frames.
		 */
		struct OnaMatch;

		typedef std::unique_ptr<OnaMatch> MatchUPtr; /**< A unique pointer to an OnaMatch */
		typedef std::shared_ptr<OnaMatch> MatchSPtr; /**< A shared pointer to an OnaMatch */
		typedef std::weak_ptr<OnaMatch> MatchWPtr; /**< A weak pointer to an OnaMatch */

		typedef std::map<int, MatchSPtr> IdMatchMapFrom; /**< Map containing matches from a frame. */
		typedef std::map<int, MatchWPtr> IdMatchMapTo; /**< Map containing matches to a frame. */

		int _id; /**< ID. */
		cv::Mat _image; /**< Image. */
		cv::Mat _cameraMatrix; /**< Camera matrix. */
		std::vector<float> _distCoeffs; /**< Radial distortion coefficients. */
		std::vector<cv::KeyPoint> _keyPoints; /**< Key points. */
		cv::Mat _descriptors; /**< Corresponding key point descriptors. */
		IdMatchMapFrom _frameMatchesFrom; /**< Matches to other frames. */
		IdMatchMapTo _frameMatchesTo; /**< Matches to this frame. */

		/**
		 * \brief Gets the OnaMatch from the id.
		 *
		 * \param id The id.
		 *
		 * \return A pointer to the match, null_ptr if none.
		 */
		MatchSPtr getMatchById(int id);

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
