/**
 * \file onamatch.hpp
 * \brief An optical navigation aid match class.
 * \author James Duley
 * \version 1.0
 * \date 2014-09-07
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

#ifndef ONAMATCH_HPP
#define ONAMATCH_HPP

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace ona {

class Frame;

class Match {
	public:
		/**
		 * \brief A pose transformation.
		 */
		struct Pose {
			cv::Mat R; /**< Rotation. */
			cv::Mat t; /**< Translation. */
		};

		typedef std::unique_ptr<Match> UPtr; /**< A unique pointer to a Match */
		typedef std::shared_ptr<Match> SPtr; /**< A shared pointer to a Match */
		typedef std::weak_ptr<Match> WPtr; /**< A weak pointer to a Match */

		/**
		 * \brief Creates a Match between Frames.
		 *
		 * \param frameToMatchFrom The query frame to match to.
		 * \param frameToMatchTo The train frame to match to.
		 * \param matcher The DescriptorMatcher to use.
		 * \param maxDistance The maximum allowable description distance to allow.
		 */
		Match(const Frame &frameToMatchFrom, const Frame &frameToMatchTo, cv::DescriptorMatcher &matcher, float maxDistance);

		/**
		 * \brief Computes the essential matrix of the match and recovers the pose difference.
		 *
		 * \param ransacMaxDistance The maximum permissible distance is pixels to epipolar line.
		 * \param ransacConfidence The desired probability that the answer is correct.
		 *
		 * \return The number of inliers.
		 */
		int compute(double ransacMaxDistance, double ransacConfidence);

		/**
		 * \brief Sets the scale from another match.
		 *
		 * \param matchFrom The match to calcualte scale from, must match a common Frame.
		 */
		void setScaleFrom(const Match &matchFrom);

		/**
		 * \brief Finds the pose difference w.r.t a frame.
		 *
		 * \return Pose transformation.
		 */
		Pose getPoseDiff();

		/**
		 * \brief Gets the 3D points.
		 *
		 * \return The 3D points.
		 */
		cv::Mat get3dPoints();

		/**
		 * \brief Draws the matched flow on an image
		 *
		 * \param image The image to use as the background.
		 *
		 * \return The image with the flow on it.
		 */
		cv::Mat drawFlow(const cv::Mat &image);

	private:
		int queryFrameId_, trainFrameId_;
		std::vector<int> queryIdx_, trainIdx_;
		std::vector<cv::Point2f> queryPoints_, trainPoints_;
		std::vector<cv::Point2f> queryNormalisedPoints_, trainNormalisedPoints_;
		cv::Mat essential_;
		std::vector<uchar> inliers_;
		Pose poseDiff_;
		cv::Mat_<float> points3D_;
		float scale_;

		/**
		 * \brief Sets the essential matrix of the match.
		 *
		 * \param ransacMaxDistance The maximum permissible distance is pixels to epipolar line.
		 * \param ransacConfidence The desired probability that the answer is correct.
		 */
		void setEssentialMatRansac(double ransacMaxDistance, double ransacConfidence);

		/**
		 * \brief Sets the pose difference for a match.
		 */
		void setPoseDiff();
};

} // namespace ona

#endif // ONAMATCH_HPP
