
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/videoio.hpp"
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <vector>
#include <iostream>
#include <cmath>



using namespace cv;
using namespace std;

#define RUN_D3_APP 1
#define SCALE_DOWN_VIDEO 0
#define DETECT_IRIS 1

#define RED_IRIS Scalar(0, 0, 255)
#define GREEN_REC Scalar(0, 255, 0)
#define WHITE_REC Scalar(255, 255, 255)

#if RUN_D3_APP

const double scale_factor = 1.2;
const int min_Neighbors = 3;
const int flags = 0 | CASCADE_SCALE_IMAGE;
const int CameraIndex = 0;
vector<Point> centers;

Rect getLeftmostEye(vector<Rect> &eyes);
Vec3f getEyeBallfromROI(Mat &eye, vector<Vec3f> &circles);
Point stabalize(vector<Point> &points, int windowsize);

int main() {
	vector<Rect> faceVect, eyeVect;
	vector<Vec3f> circles;
	int circleCount = 0;
	Vec3f eyeball;

	Mat frame, gray_face, leftEye;//, eye;

	Rect eyeRect;


	CascadeClassifier face_cascade , eye_cascade, roi_face, roi_color;
	face_cascade = cv::CascadeClassifier("haarcascade_frontalface_default.xml");
	eye_cascade = cv::CascadeClassifier("haarcascade_eye.xml");


	cv::VideoCapture camera(CameraIndex); // Try opening camera

#if SCALE_DOWN_VIDEO
	camera.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
#endif

	camera.set(CAP_PROP_FPS, 60);
	if (!camera.isOpened())
	{
		fprintf(stderr, "Error - Camera could not be accessed\n");
		exit(1);
	}

	while (true)
	{
		camera.read(frame);
		cv::flip(frame, frame, 1); //flip the frame horizontally

		if (waitKey(30) == 27)
		{
			return 0;
		}
		cvtColor(frame, gray_face, COLOR_BGR2GRAY, 0);
		equalizeHist(gray_face, gray_face);
		face_cascade.detectMultiScale(gray_face, faceVect, scale_factor, min_Neighbors, flags, Size(100, 100));//, Size(250, 250)); //, 1.05, 3, 0|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 30), cv::Size(80, 80));
		Mat face_roi = gray_face(faceVect[0]);
		//for (uint64_t i = 0; i < faceVect.size(); i++) //detecting the face
		//{
			rectangle(frame, faceVect[0], GREEN_REC, 1, 8, 0);
			eye_cascade.detectMultiScale(face_roi, eyeVect, scale_factor, min_Neighbors, flags, Size(30, 30), Size(70, 70));
			for(uint64_t j = 0; j < eyeVect.size(); j++) //detecting the eyes
			{

				//Point center( faceVect[i].x + eyeVect[j].x + eyeVect[j].width*0.5, faceVect[i].y + eyeVect[j].y + eyeVect[j].height*0.5 );
				//int radius = cvRound( (eyeVect[j].width + eyeVect[j].height)*0.25 );
				//circle( frame, center, radius, Scalar( 255, 0, 0 ), 1, 8, 0 );
				if(eyeVect.size() != 2)
				{
					//printf("both eyes not found\n");
					break; //if both eyes are not found, return out.
				}
				rectangle(frame, faceVect[0].tl() + eyeVect[j].tl(), faceVect[0].tl() + eyeVect[j].br(), WHITE_REC, 2);
				//rectangle(frame, cvPoint(eyeVect[j].x, eyeVect[j].y), cvPoint(+ eyeVect[j].width + eyeVect[j].x
				//		,eyeVect[j].height + eyeVect[j].y), Scalar(0, 0, 255), 1, 8, 0);

				//printf("BEFORE call to getLeftmostEye\n");
				eyeRect = getLeftmostEye(eyeVect);
				Mat eye = face_roi(eyeRect);
				equalizeHist(eye, eye);
#if DETECT_IRIS
				HoughCircles(eye, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 6, eye.rows / 4);
				//rectangle(frame, faceVect[0] + eyeVect[j].tl(), Scalar(255, 0, 0), 1, 8, 0);
				if(circles.size() > 0)
				{
					printf("Circles > %i\n", circleCount);
					eyeball = getEyeBallfromROI(eye, circles);
					Point center(eyeball[0], eyeball[1]);
					centers.push_back(center);
					center = stabalize(centers, 5);
					int radius = (int)eyeball[2];
					circle(frame, faceVect[0].tl() + eyeRect.tl() + center, radius, RED_IRIS, 1);
					//circle(eye, center, radius, Scalar(0, 0, 255), 2);
					circleCount++;

				}
#endif
			}
		//}
		imshow("Distracted Driver Detection", frame);
	}
	return 0;
}

Rect getLeftmostEye(vector<Rect> &eyes)
{
	int leftmost = 99999999;
	int leftmostIndex = -1;
	for(uint64_t i = 0; i < eyes.size(); i++)
	{
		if(eyes[i].tl().x < leftmost){
			leftmost = eyes[i].tl().x;
			leftmostIndex = i;
		}
	}
	return eyes[leftmostIndex];

}

Vec3f getEyeBallfromROI(Mat &eye, vector<Vec3f> &circles)
{
	int radius;
	vector<int> sums(circles.size(), 0);
	for(int x = 0; x < eye.rows; x++)
	{
		uchar *ptr = eye.ptr<uchar>(x);
		for(int y = 0; y < eye.cols; y++)
		{
			int value = static_cast<int>(*ptr);
			for(uint64_t i = 0; i < circles.size(); i++)
			{
				Point center((int)round(circles[i][0]), (int)round(circles[i][1]));
				radius = (int)round(circles[i][2]);
				if(pow(y - center.y, 2 + pow(x - center.x, 2) < pow(radius, 2)))
				{
					sums[i] += value;
				}
			}
			++ptr;
		}
	}
	int smallestSum = 99999999;
	int smallestSumIndex = -1;
	for(uint64_t i = 0; i < circles.size(); i++)
	{
		if(sums[i] < smallestSum)
		{
			smallestSum = sums[i];
			smallestSumIndex = i;
		}
	}
	return circles[smallestSumIndex];
}
Point stabalize(vector<Point> &points, int windowsize)
{
	float sumX = 0;
	float sumY = 0;
	int count = 0;
	for(uint64_t i = max(0, (int)(points.size() - windowsize)); i < points.size(); i++)
	{
		sumX += points[i].x;
		sumY += points[i].y;
		++count;
	}
	if(count > 0)
	{
		sumX /= count;
		sumY /= count;
	}
	return Point(sumX, sumY);
}
#endif





