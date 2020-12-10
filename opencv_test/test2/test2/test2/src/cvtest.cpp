#include<opencv2/opencv.hpp>

int main(void)
{
	cv::Mat red_img(cv::Size(640, 480), CV_8UC3, cv::Scalar(0, 0, 255));
	cv::rectangle(red_img, cv::Point(200, 350), cv::Point(300, 450), cv::Scalar(200, 0, 0), -1);

	cv::imwrite("cvtest.jpg", red_img);
	return 0;
}