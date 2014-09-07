/**
 * \file pose.hpp
 * \author James Duley
 * \version 1.0
 */

#ifndef POSE_HPP
#define POSE_HPP


#include <opencv2/core/core.hpp>

/**
 * \brief Decomposes the essential matirx into the rotation and translation.
 *
 * \param E The essential matrix to decompose.
 * \param R The rotation matrix.
 * \param t The translation of unit scale.
 */
void getRt(const cv::Mat_<double> &E, cv::Mat_<double> &R, cv::Mat_<double> &t);

/**
 * \brief Composes the camera projection matrix.
 *
 * \param P The camera projection matrix.
 * \param R The roation matrix.
 * \param t The translation matrix.
 */
void getP(cv::Matx34d &P, const cv::Mat_<double> &R, const cv::Mat_<double> &t);

/**
 * \brief Check whether the rotation has unit determinant.
 *
 * \param R The rotation matrix.
 *
 * \return True if the matrix is coherent.
 */
bool checkCoherentRotation(const cv::Mat_<double> &R);

/**
 * \brief Print the Euler angles of a rotation matrix.
 *
 * \param R The rotation matrix.
 * \param angles The Euler angles in degrees.
 */
void getEulerAngles(const cv::Mat_<double> &R, cv::Vec3d &angles);


namespace cv {

/**
 * \brief Recovers the pose froman essential matrix
 *
 * \param E The essential matrix.
 * \param points1 The first set of points used for E.
 * \param points2 The second set of points used for E.
 * \param R The rotation.
 * \param t The translation with unit magnitude.
 * \param U The triangulated points in homogenous coordinates scaled with the translation.
 * \param mask Inliers.
 *
 * \return 
 */
int recoverPose( cv::InputArray E, cv::InputArray points1, cv::InputArray points2,
                            cv::OutputArray R, cv::OutputArray t,
			    cv::OutputArray U,
                            cv::InputOutputArray mask = cv::noArray() );
}

#endif
