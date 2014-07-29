
#include "pose.hpp"

#include <iostream>

using namespace cv;

void getRt(const Mat_<double> &E, Mat_<double> &R, Mat_<double> &t) {

	// SVD to get projection
	Mat w, u, vt;
	SVD::compute(E, w, u, vt, SVD::MODIFY_A);

	Matx33d W(0., -1., 0., 1., 0., 0., 0., 0., 1.);

	R = u * Mat(W) * vt;
	
	t = u.col(2);
}

void getP(Matx34d &P, const Mat_<double> &R, const Mat_<double> &t) {
	P = Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0), 
			R(1, 0), R(1, 1), R(1, 2), t(1), 
			R(2, 0), R(2, 1), R(2, 2), t(2));
}

bool checkCoherentRotation(const Mat_<double> &R) {
	if (std::abs(determinant(R)) - 1. > 1e-07) {
		std::cerr << "|det(R)| != 1, not a rotation matrix" << std::endl;
		return false;
	}

	return true;
}

void printEulerAngles(const Mat_<double> &R) {
	double alpha = std::atan2(R(2, 1), R(2, 2));
	double beta = std::atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
	double gamma = std::atan2(R(1, 0), R(0, 0));

	std::cout << alpha << " " << beta << " " << gamma << " " << std::endl;
}

