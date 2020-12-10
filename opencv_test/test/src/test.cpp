# include <opencv2/opencv.hpp>
# include <opencv2/highgui.hpp>
# include <opencv2/imgproc/imgproc_c.h> //CV_AA用

void main() {
	cv::namedWindow("test",cv::WINDOW_NORMAL);

	cv::Mat img = cv::Mat::zeros(500, 500, CV_8UC3); //500×500ピクセルの黒色のMat

	rectangle(img, cv::Point(25, 180), cv::Point(450, 300), cv::Scalar(0, 255, 0), CV_FILLED, 8, 0);
	putText(img, "TEST OPENCV", cv::Point(20, 250), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 0), 5, CV_AA);

	imshow("test", img);
	cv::waitKey(0);
	return;
}