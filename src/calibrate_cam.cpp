#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 

using namespace cv;
 
int main() { 
	VideoCapture capture(0);// connect to the camera 
	Mat image;
	capture.read(image); // get the first frame of video 

	while (waitKey(10) < 0) { 
		capture.read(image); // get the first frame of video 
		imshow("cam image", image); // display image 
	} 

	return 0; 
}
