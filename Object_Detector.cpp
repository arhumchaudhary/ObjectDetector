// OpenCV_HelloWorld.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// OpenCV Imports
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> // OpenCV Core Functionality
#include <opencv2/highgui/highgui.hpp> // High-Level Graphical User Interface
#include <list>


// [Optional] Use OpenCV namespace

using namespace cv;
using namespace std;


const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.04f;


Mat loadImage(String img) {
	Mat colourImg = imread(img);
	Mat greyImg;
	cvtColor(colourImg, greyImg, COLOR_BGR2GRAY);
	return greyImg;
}


#define M_PI 3.14159265358979323846

Mat getSimilarityMatrix(std::vector<Point2f> ptsA, std::vector<Point2f> ptsB, float& scale, float& theta, float& tx, float& ty)
{
	int nPts = ptsA.size();
	int rows = 2 * nPts, cols = 5;
	Mat A(rows, cols, CV_32FC1);

	for (int i = 0; i < nPts; i++)
	{
		A.at<float>(2 * i, 0) = ptsA[i].x;
		A.at<float>(2 * i, 1) = -ptsA[i].y;
		A.at<float>(2 * i, 2) = 1.0;
		A.at<float>(2 * i, 3) = 0.0;
		A.at<float>(2 * i, 4) = -ptsB[i].x;

		A.at<float>(2 * i + 1, 0) = ptsA[i].y;
		A.at<float>(2 * i + 1, 1) = ptsA[i].x;
		A.at<float>(2 * i + 1, 2) = 0.0;
		A.at<float>(2 * i + 1, 3) = 1.0;
		A.at<float>(2 * i + 1, 4) = -ptsB[i].y;
	}

	SVD svd;
	Mat X;
	svd.solveZ(A, X);

	float sa = X.at<float>(0) / X.at<float>(4);
	float sb = X.at<float>(1) / X.at<float>(4);
	tx = X.at<float>(2) / X.at<float>(4);
	ty = X.at<float>(3) / X.at<float>(4);

	scale = (float)sqrt((double)(sa * sa + sb * sb));
	theta = atan2(sb, sa) * 180.0 / M_PI;

	Mat M(3, 3, CV_32FC1);
	M.at<float>(0, 0) = sa;
	M.at<float>(0, 1) = -sb;
	M.at<float>(0, 2) = tx;
	M.at<float>(1, 0) = sb;
	M.at<float>(1, 1) = sa;
	M.at<float>(1, 2) = ty;
	M.at<float>(2, 0) = 0;
	M.at<float>(2, 1) = 0;
	M.at<float>(2, 2) = 1;

	return(M);
}


int main(int argc, char** argv) {

	Mat imgA = imread("painting.png");
	Mat imgB = imread("paintingbkd.png");


	int nfeatures = 1000;
	float scaleFactor = 1.2f;
	int nlevels = 8;
	int edgeThreshold = 31;
	int firstLevel = 0;
	int WTA_K = 2;
	ORB::ScoreType scoreType = ORB::HARRIS_SCORE;
	int patchSize = 31;

	Ptr<FeatureDetector> detector = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);

	Ptr<DescriptorExtractor> descriptor = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize);

	vector<KeyPoint> kp_object, kp_scene;
	detector->detect(imgA, kp_object);
	detector->detect(imgB, kp_scene);

	Mat descriptors_obj, descriptors_scene;
	descriptor->compute(imgA, kp_object, descriptors_obj);
	descriptor->compute(imgB, kp_scene, descriptors_scene);


	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	// Create a vector of matches
	vector<DMatch> matches;
	// Match the descriptors
	matcher->match(descriptors_obj, descriptors_scene, matches);

	//only take good matches
	std::sort(matches.begin(), matches.end());
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());

	Mat output;

	drawMatches(imgA, kp_object, imgB, kp_scene, matches, output);

	std::vector<Point2f> obj;
	std::vector<Point2f> scene;


	float scale, theta, tx, ty;
	std::vector < cv::Point2f > ptsA, ptsB;

	//to get the points from the "matches"
	for (int i = 0; i < (int)matches.size(); i++)
	{
		ptsA.push_back(kp_object[matches[i].queryIdx].pt);
		ptsB.push_back(kp_scene[matches[i].trainIdx].pt);
	}


	srand(time(NULL));
	int r_idx;
	int r_idx2;

	int k = 4000;
	int m = 4;
	int t = 90;
	int cnt = 0;
	float p;
	float p_th = 0.1;


	Mat bestMatrix;
	int bestInlier = 0;

	std::vector < cv::Point2f > ptsA1, ptsB1, ptsInlier;
	double dist;
	std::vector < cv::Point2f> ptsApro;

	Mat H = findHomography(ptsA, ptsB, RANSAC);
	while (cnt < k) {

		cnt++;
		int inlier = 0;
		ptsInlier.clear();

		for (int i = 0; i < m; i++) {
			r_idx = rand() % (matches.size());
			ptsA1.push_back(ptsA[r_idx]);
			ptsB1.push_back(ptsB[r_idx]);
		}


		cv::Mat mySimilarityMatrix = getSimilarityMatrix(ptsA1, ptsB1, scale, theta, tx, ty);
		cv::perspectiveTransform(ptsA, ptsApro, mySimilarityMatrix);


		for (int i = 0; i < ptsApro.size(); i++) {
			dist = sqrt(pow(ptsApro[i].x - ptsB[i].x, 2) + (pow(ptsApro[i].y - ptsB[i].y, 2)));

			if (dist < t) {
				inlier++;
			}

		}

		p = inlier / float(ptsApro.size());
		//cout << p << endl;
		if (p > p_th) {
			//bestInlier = inlier;
			p_th = p;
			bestMatrix = mySimilarityMatrix;
			cout << "Better M:" << bestMatrix << endl;
		}

		if (cnt == k / 3)
			cout << "1/3 done" << endl;
		else if (cnt == k / 2)
			cout << "50% done" << endl;
		else if (cnt == 0.75 * k)
			cout << "75% done" << endl;
		else if (cnt == 0.9 * k)
			cout << "90% done" << endl;
	}

	cout << H << endl;
	cout << bestMatrix<< endl;

	cv::Point pt1(0, 0);
	cv::Point pt2(imgA.cols, imgA.rows);
	Mat blueRect = Mat::zeros(imgA.size(),CV_8UC3);
	
	cv::rectangle(blueRect, pt1, pt2, cv::Scalar(255, 0, 0),-1); 

	Mat img_aligned = Mat(imgB.size(),CV_8UC3);
	warpPerspective(blueRect, img_aligned, H, imgB.size(), INTER_LINEAR);

	Mat newImgB = imgB.clone();

	cv::addWeighted(img_aligned, 0.2, imgB, 0.8, 0.0, newImgB);

	Mat img_aligned2 = Mat(imgB.size(), CV_8UC3);
	warpPerspective(blueRect, img_aligned2, bestMatrix, imgB.size(), INTER_LINEAR);

	Mat newImgB2 = imgB.clone();

	cv::addWeighted(img_aligned2, 0.2, imgB, 0.8, 0.0, newImgB2);
	

	namedWindow("Image A", cv::WINDOW_AUTOSIZE);
	imshow("Image A", imgA);
	namedWindow("Image B", cv::WINDOW_AUTOSIZE);
	imshow("Image B", imgB);
	namedWindow("Blue Rect", cv::WINDOW_AUTOSIZE);
	imshow("Blue Rect", blueRect); 
	namedWindow("aligned", cv::WINDOW_AUTOSIZE);
	imshow("aligned", img_aligned);
	namedWindow("overlay", cv::WINDOW_AUTOSIZE);
	imshow("overlay", newImgB);
	namedWindow("overlay2", cv::WINDOW_AUTOSIZE);
	imshow("overlay2", newImgB2);
	namedWindow("Combined", cv::WINDOW_AUTOSIZE);
	imshow("Combined", output);
    
    	imwrite("/result.png", newImgB2);

	cv::waitKey();
	return 0;
}
