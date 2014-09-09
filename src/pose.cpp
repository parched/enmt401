
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

/*-----from opencv 3 five-point.cpp------------------------------*/
namespace cv {
void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t )
{
    Mat E = _E.getMat().reshape(1, 3);
    CV_Assert(E.cols == 3 && E.rows == 3);

    Mat D, U, Vt;
    SVD::compute(E, D, U, Vt);

    if (determinant(U) < 0) U *= -1.;
    if (determinant(Vt) < 0) Vt *= -1.;

    Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    W.convertTo(W, E.type());

    Mat R1, R2, t;
    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;
    t = U.col(2) * 1.0;

    R1.copyTo(_R1);
    R2.copyTo(_R2);
    t.copyTo(_t);
}

int recoverPose( InputArray E, InputArray _points1, InputArray _points2, OutputArray _R,
                     OutputArray _t, OutputArray _U, InputOutputArray _mask)
{
    Mat points1, points2;
    _points1.getMat().copyTo(points1);
    _points2.getMat().copyTo(points2);

    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
                              points1.type() == points2.type());

    if (points1.channels() > 1)
    {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }
    points1.convertTo(points1, CV_64F);
    points2.convertTo(points2, CV_64F);

    points1 = points1.t();
    points2 = points2.t();

    Mat R1, R2, t;
    decomposeEssentialMat(E, R1, R2, t);
    Mat P0 = Mat::eye(3, 4, R1.type());
    Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
    P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
    P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
    P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.
    double dist = 50.0;
    Mat Q;
    Mat U1;
    triangulatePoints(P0, P1, points1, points2, U1);
    Mat mask1 = U1.row(2).mul(U1.row(3)) > 0;
    U1.row(0) /= U1.row(3);
    U1.row(1) /= U1.row(3);
    U1.row(2) /= U1.row(3);
    U1.row(3) /= U1.row(3);
    mask1 = (U1.row(2) < dist) & mask1;
    Q = P1 * U1;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < dist) & mask1;

    Mat U2;
    triangulatePoints(P0, P2, points1, points2, U2);
    Mat mask2 = U2.row(2).mul(U2.row(3)) > 0;
    U2.row(0) /= U2.row(3);
    U2.row(1) /= U2.row(3);
    U2.row(2) /= U2.row(3);
    U2.row(3) /= U2.row(3);
    mask2 = (U2.row(2) < dist) & mask2;
    Q = P2 * U2;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < dist) & mask2;

    Mat U3;
    triangulatePoints(P0, P3, points1, points2, U3);
    Mat mask3 = U3.row(2).mul(U3.row(3)) > 0;
    U3.row(0) /= U3.row(3);
    U3.row(1) /= U3.row(3);
    U3.row(2) /= U3.row(3);
    U3.row(3) /= U3.row(3);
    mask3 = (U3.row(2) < dist) & mask3;
    Q = P3 * U3;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < dist) & mask3;

    Mat U4;
    triangulatePoints(P0, P4, points1, points2, U4);
    Mat mask4 = U4.row(2).mul(U4.row(3)) > 0;
    U4.row(0) /= U4.row(3);
    U4.row(1) /= U4.row(3);
    U4.row(2) /= U4.row(3);
    U4.row(3) /= U4.row(3);
    mask4 = (U4.row(2) < dist) & mask4;
    Q = P4 * U4;
    mask4 = (Q.row(2) > 0) & mask4;
    mask4 = (Q.row(2) < dist) & mask4;

    // If _mask is given, then use it to filter outliers.
    if (!_mask.empty())
    {
        Mat mask = _mask.getMat().reshape(1, 1);
        CV_Assert(mask.size() == mask1.size());
        bitwise_and(mask, mask1, mask1);
        bitwise_and(mask, mask2, mask2);
        bitwise_and(mask, mask3, mask3);
        bitwise_and(mask, mask4, mask4);
    }
    if (_mask.empty() && _mask.needed())
    {
        _mask.create(mask1.size(), CV_8U);
    }

    CV_Assert(_R.needed() && _t.needed());
    _R.create(3, 3, R1.type());
    _t.create(3, 1, t.type());

    int good1 = countNonZero(mask1);
    int good2 = countNonZero(mask2);
    int good3 = countNonZero(mask3);
    int good4 = countNonZero(mask4);

    if (good1 >= good2 && good1 >= good3 && good1 >= good4)
    {
        R1.copyTo(_R);
        t.copyTo(_t);
	if (_U.needed()) U1.copyTo(_U);
        if (_mask.needed()) mask1.copyTo(_mask);
        return good1;
    }
    else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
    {
        R2.copyTo(_R);
        t.copyTo(_t);
	if (_U.needed()) U2.copyTo(_U);
        if (_mask.needed()) mask2.copyTo(_mask);
        return good2;
    }
    else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
    {
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
	if (_U.needed()) U3.copyTo(_U);
        if (_mask.needed()) mask3.copyTo(_mask);
        return good3;
    }
    else
    {
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
	if (_U.needed()) U4.copyTo(_U);
        if (_mask.needed()) mask4.copyTo(_mask);
        return good4;
    }
}
}

