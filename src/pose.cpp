
#include "pose.hpp"

#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

void getRt(const cv::Mat_<double> &E, cv::Mat_<double> &R, cv::Mat_<double> &t) {

	// SVD to get projection
	cv::Mat w, u, vt;
	cv::SVD::compute(E, w, u, vt, cv::SVD::MODIFY_A);

	cv::Matx33d W(0., -1., 0., 1., 0., 0., 0., 0., 1.);

	R = u * cv::Mat(W) * vt;
	
	t = u.col(2);
}

void getP(cv::Matx34d &P, const cv::Mat_<double> &R, const cv::Mat_<double> &t) {
	P = cv::Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0), 
			R(1, 0), R(1, 1), R(1, 2), t(1), 
			R(2, 0), R(2, 1), R(2, 2), t(2));
}

bool checkCoherentRotation(const cv::Mat_<double> &R) {
	if (std::abs(cv::determinant(R)) - 1. > 1e-07) {
		std::cerr << "|det(R)| != 1, not a rotation matrix" << std::endl;
		return false;
	}

	return true;
}

void getEulerAngles(const cv::Mat_<double> &R, cv::Vec3d &angles) {
	angles(0)  = std::atan2(R(2, 1), R(2, 2)) * 180.0 / CV_PI;
	angles(1)  = std::atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2))) * 180.0 / CV_PI;
	angles(2)  = std::atan2(R(1, 0), R(0, 0)) * 180.0 / CV_PI;
}
