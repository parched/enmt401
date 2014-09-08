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

namespace ona {

/**
 * \brief An optical navgiation aid frame.
 */
class Frame {
	friend class Match;

	public:
		typedef std::unique_ptr<Frame> UPtr; /**< A unique pointer to a Frame */
		typedef std::shared_ptr<Frame> SPtr; /**< A shared pointer to a Frame */
		typedef std::weak_ptr<Frame> WPtr; /**< A weak pointer to a Frame */

		/**
		 * \brief Frame constructor.
		 *
		 * \param id The id of the frame which should be unique.
		 * \param image The frame image.
		 * \param cameraMatrix The corresponding camera matirx.
		 * \param distCoeffs The corresponding radial distortion coefficients.
		 */
		Frame(int id, const cv::Mat &image, const cv::Mat &cameraMatrix, const std::vector<float> &distCoeffs);

		/**
		 * \brief Gets the id of the Frame
		 *
		 * \return The id.
		 */
		int getId() const;

		/**
		 * \brief Gets the image of the Frame.
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

	private:
		int id_; /**< ID. */
		cv::Mat image_; /**< Image. */
		cv::Mat cameraMatrix_; /**< Camera matrix. */
		std::vector<float> distCoeffs_; /**< Radial distortion coefficients. */
		std::vector<cv::KeyPoint> keyPoints_; /**< Key points. */
		cv::Mat descriptors_; /**< Corresponding key point descriptors. */
};

} // namespace ona

#endif
