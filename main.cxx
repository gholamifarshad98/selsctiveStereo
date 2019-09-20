#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include<math.h>
#include<vector>
#include<memory>
#include <chrono> 
#include <string> 
using namespace cv;
using namespace std;
using namespace std::chrono;
struct pixel
{
	int row;
	int column;
	int disparity;
	bool consistance;
};
struct Stain
{
	int i;  // center of aera of stain in x direction.
	int j;  // center of aera of stain in y direction.
	int minI; // boundry of stain in x direction.
	int maxI;
	int minJ; // boundry of stain in y direction.
	int maxJ;
	vector<Point> stainPoints;

};
int numOfColumns;
int numOfRows;
int thickness = 60;
int maxDisparity = 30;
int maxkernelSize = 35; // kernel size must be odd number.
int kernelSize = 5;
int midleDisparity = 14;
auto sstereoResult = make_shared<Mat>();
typedef vector<pixel*> layerVector;
vector<layerVector> layers;
void ReadBothImages(shared_ptr<Mat>, shared_ptr<Mat>);
void Meshing(int, int, int, int, int);
double CalcDistance(int, int, int, int);
int CalcCost(shared_ptr<Mat>, shared_ptr<Mat>, int, int, int, int);
Vec3b bgrPixel_02(0, 255, 255);
Vec3b bgrPixel_04(255, 0, 0);
Vec3b bgrPixel_03(0, 255, 0);
Vec3b bgrPixel_01(0, 0, 255);
Vec3b bgrBackground(0, 0, 0);

void stereo(shared_ptr<Mat>, shared_ptr<Mat>, layerVector*, int, int);
void selsectiveStereo(shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, layerVector*, int, int);
void selsectiveStereo(shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, int, int);
void prepareResult(shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Mat>, vector<layerVector>, int, int, int, string);
void filterResult(shared_ptr<Mat>, shared_ptr<Mat>, Vec3b);
void checkPoint(shared_ptr<Mat>, shared_ptr<Mat>, shared_ptr<Stain>, int, int, Vec3b);
void makeStain(shared_ptr<Mat> , shared_ptr<Mat> , shared_ptr<Stain> , int , int, Vec3b);
void stainDetector(shared_ptr<Mat>, shared_ptr<Mat>, Vec3b);
int main()
{
	auto rightImage = make_shared<Mat>();
	auto leftImage = make_shared<Mat>();
	ReadBothImages(leftImage, rightImage);
	Meshing(numOfRows, numOfColumns, thickness, maxkernelSize, maxDisparity);
	auto start = chrono::high_resolution_clock::now();
	try
	{
		for (int i = 0; i < layers.size(); i++) {
			stereo(leftImage, rightImage, &layers[i], kernelSize, maxDisparity);
		}
	}
	catch (cv::Exception & e)
	{
		cerr << e.msg << endl; // output exception message
	}
	chrono::high_resolution_clock::time_point stop = high_resolution_clock::now();
	auto duration = duration_cast<seconds>(stop - start);
	auto value = duration.count();
	string duration_s = to_string(value);
	for (int midDis = 1; midDis < maxDisparity; midDis++) {
		auto result_00 = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC1);// Stereo result.
		auto result_01 = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC3);// Selective stereo L2R.
		auto result_02 = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC3);// Selective stereo R2L.
		auto result_03 = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC3);// slective with L2R and R2L consistance.
		auto result_04 = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC3);// slective with L2R and R2L notconsistance.
		auto result_total = make_shared<Mat>(4 * numOfRows, numOfColumns, CV_8UC3, bgrBackground);
		prepareResult(result_00, result_01, result_02, result_03, layers, numOfRows, numOfColumns, kernelSize, duration_s);

		////////////////////////////////////////////////////////////////////
		/// In this part we have impelemet selective stereo.
		////////////////////////////////////////////////////////////////////
		selsectiveStereo(leftImage, rightImage, result_01, result_02, result_03, kernelSize, midDis);

		try {
			cvtColor(*result_00, *result_00, CV_GRAY2RGB);
			result_00->copyTo((*result_total)(Rect(0, 0 * numOfRows, numOfColumns, numOfRows)));
			result_01->copyTo((*result_total)(Rect(0, 1 * numOfRows, numOfColumns, numOfRows)));
			result_02->copyTo((*result_total)(Rect(0, 2 * numOfRows, numOfColumns, numOfRows)));
			result_03->copyTo((*result_total)(Rect(0, 3 * numOfRows, numOfColumns, numOfRows)));

		}
		catch (cv::Exception & e)
		{
			cerr << e.msg << endl; // output exception message
		}
		string temp;
		temp = "result_midDis_" + to_string(midDis) + "withFilter.png";
		imwrite(temp, *result_03);
		filterResult(result_00, result_03, bgrPixel_04);
		temp = "result_midDis_" + to_string(midDis) + "withoutFilter.png";
		imwrite(temp, *result_03);
		imshow("result_total", *result_03);
		stainDetector(result_00, result_03, bgrPixel_04);
		waitKey(1000);
		cout << midDis << endl;
	}
	return 0;
}

////////////////////////////////////////////////////////////////////
/// In this part we can load two Images.
////////////////////////////////////////////////////////////////////
void ReadBothImages(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage) {

	try {
		*rightImage = imread("000147_11.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the right image
		//rightImage->convertTo(*rightImage, CV_64F);
		*rightImage = *rightImage;
		*leftImage = imread("000147_10.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the left image
		//leftImage->convertTo(*leftImage, CV_64F);
		*leftImage = *leftImage;
		numOfRows = leftImage->rows;
		numOfColumns = leftImage->cols;
		if (!rightImage->data)                             // Check for invalid input
		{
			throw "right";
		}
		if (!leftImage->data)                             // Check for invalid input
		{
			throw "left";
		}
	}
	catch (char* error) {
		cout << "can not load the " << error << " iamge" << endl;
	}
	//imshow("test", *rightImage);

	//waitKey(0);
}


////////////////////////////////////////////////////////////////////
/// In this part we clac layer of each pixel.
////////////////////////////////////////////////////////////////////
void Meshing(int numOfRows, int numOfColumns, int thickness, int kernelSize, int maxDisparity) {
	int tempLayer = 0;
	int numOfLayers = int(CalcDistance(numOfRows, numOfColumns, 0, 0) / thickness);
	// the number 4 thai wrote there is for ensure that all of the image has suported... dont wworry... we have delete those who is null.
	for (int i = 1; i <= numOfLayers + 4; i++) {
		layerVector tempLayer;
		layers.push_back(tempLayer);
	}
	for (int i = (kernelSize / 2); i < numOfRows - (kernelSize / 2); i++) {
		for (int j = (kernelSize / 2); j < numOfColumns - (kernelSize / 2) - maxDisparity; j++) {
			tempLayer = int(CalcDistance(numOfRows, numOfColumns, i, j) / thickness);
			pixel* tempLocation = new pixel;
			tempLocation->row = i;
			tempLocation->column = j;
			layers.at(tempLayer).push_back(tempLocation);
		}
	}
	// this part is added to avoid vector with zeero size.
	for (int i = layers.size() - 1; i >= 0; i = i - 1) {
		if (layers[i].size() == 0) {
			layers.erase(layers.begin() + i);
		}
	}
}


////////////////////////////////////////////////////////////////////
/// In this part we clac distance of each pixel.
////////////////////////////////////////////////////////////////////
double CalcDistance(int numOfRows, int numOfColumns, int row, int column) {
	double tempDistance = sqrt(pow((row - numOfRows), 2) + pow((column - int(numOfColumns / 2) + .05), 2));
	return tempDistance;
}


////////////////////////////////////////////////////////////////////
/// In this part we clac disparity of each pixel.
////////////////////////////////////////////////////////////////////
void stereo(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage, layerVector* layer, int kernelSize, int maxDisparity) {
	//imshow("leftImage", *leftImage);
	//imshow("rightImage", *rightImage);
	//waitKey(12);
	int tempCost = 0;
	int tempDisparity = 0;
	for (int p = 1; p < layer->size(); p++) {
		//cout << "the alye size is " << layer->size() << endl;
		//cout << "disparity is  " << p << endl;
		double cost = 10000000;
		tempCost = 0;
		tempDisparity = 0;
		for (int i = 0; i < maxDisparity; i++) {
			tempCost = CalcCost(leftImage, rightImage, (*layer)[p]->row, (*layer)[p]->column, kernelSize, i);
			if (tempCost < cost) {
				cost = tempCost;
				tempDisparity = i;
			}
		}
		(*layer)[p]->disparity = tempDisparity;
	}
}


////////////////////////////////////////////////////////////////////
/// In this part we clac cost of each pixel for sepecfic disparity.
////////////////////////////////////////////////////////////////////
int CalcCost(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage, int row, int column, int kernelSize, int disparity) {
	int cost = 0;
	for (int u = -int(kernelSize / 2); u <= int(kernelSize / 2); u++) {
		for (int v = -int(kernelSize / 2); v <= int(kernelSize / 2); v++) {
			int temp1 = row + u;
			int temp2 = column + v;
			int temp3 = row + u + disparity;
			int temp4 = column + v;
			// for error handeling.
			if (column + u + disparity > numOfColumns) {
				cout << "*****************************************************" << endl;
			}
			cost = cost + int(pow((leftImage->at<uchar>(row + v, column + u) - (rightImage->at<uchar>(row + v, column + u + disparity))), 2));
		}
	}
	return cost;
}


////////////////////////////////////////////////////////////////////
/// In this part we can make the result.
////////////////////////////////////////////////////////////////////
void prepareResult(shared_ptr<Mat> result, shared_ptr<Mat> result_01, shared_ptr<Mat> result_02, shared_ptr<Mat> result_03, vector<layerVector> layers, int numOfRows, int numOfColumns, int kernalSize, string Dutime) {

	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i].size(); j++) {
			result->at<uchar>(layers[i][j]->row, layers[i][j]->column) = uchar(255 * layers[i][j]->disparity / 30);
		}

	}
	//string temp;
	//auto result_temp = make_shared<Mat>(numOfRows, numOfColumns, CV_8UC3);
	//temp = "result_KernelSize_" + to_string(kernalSize) + "_MaxDisparity_" + to_string(maxDisparity) + "Time_" + Dutime + "s.png";
	cvtColor(*result, *result_01, CV_GRAY2RGB);
	cvtColor(*result, *result_02, CV_GRAY2RGB);
	cvtColor(*result, *result_03, CV_GRAY2RGB);
}


////////////////////////////////////////////////////////////////////
/// In this part we clac selective disparity of each pixel.
////////////////////////////////////////////////////////////////////
void selsectiveStereo(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage, shared_ptr<Mat> result_1, shared_ptr<Mat> result_2, shared_ptr<Mat> result_3, layerVector* layer, int kernelSize, int midelDisparity) {
	bool left2right = false;
	bool right2let = false;
	int temp0 = midelDisparity - 1;
	int temp1 = midelDisparity;
	int temp2 = midelDisparity + 1;

	int temp3 = -(midelDisparity - 1);
	int temp4 = -midelDisparity;
	int temp5 = -(midelDisparity + 1);

	int tempCost0;
	int tempCost1;
	int tempCost2;

	int tempCost3;
	int tempCost4;
	int tempCost5;
	for (int p = 1; p < layer->size(); p++) {
		left2right = false;
		right2let = false;
		tempCost0 = CalcCost(leftImage, rightImage, (*layer)[p]->row, (*layer)[p]->column, kernelSize, temp0);
		tempCost1 = CalcCost(leftImage, rightImage, (*layer)[p]->row, (*layer)[p]->column, kernelSize, temp1);
		tempCost2 = CalcCost(leftImage, rightImage, (*layer)[p]->row, (*layer)[p]->column, kernelSize, temp2);

		tempCost3 = CalcCost(rightImage, leftImage, (*layer)[p]->row, (*layer)[p]->column, kernelSize, temp3);
		tempCost4 = CalcCost(rightImage, leftImage, (*layer)[p]->row, (*layer)[p]->column, kernelSize, temp4);
		tempCost5 = CalcCost(rightImage, leftImage, (*layer)[p]->row, (*layer)[p]->column, kernelSize, temp5);

		if (tempCost1 < tempCost0 & tempCost1 < tempCost2) {
			left2right = true;
			result_1->at<Vec3b>(Point((*layer)[p]->column, (*layer)[p]->row)) = bgrPixel_01;
		}
		if (tempCost4 < tempCost3 & tempCost4 < tempCost5) {
			right2let = true;
			result_2->at<Vec3b>(Point((*layer)[p]->column, (*layer)[p]->row)) = bgrPixel_02;
		}
		if ((left2right & right2let)) {

			result_3->at<Vec3b>(Point((*layer)[p]->column, (*layer)[p]->row)) = bgrPixel_04;
		}


	}
}

void selsectiveStereo(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage, shared_ptr<Mat> result1, shared_ptr<Mat> result2, shared_ptr<Mat> result3, int kernelSize, int midelDisparity) {
	for (int i = 0; i < layers.size(); i++) {
		selsectiveStereo(leftImage, rightImage, result1, result2, result3, &layers[i], kernelSize, midelDisparity);
	}
}

////////////////////////////////////////////////////////////////////
/// In this part we will Filter the result.
////////////////////////////////////////////////////////////////////
void filterResult(shared_ptr<Mat> background, shared_ptr<Mat> input, Vec3b Color) {
	int numberOfHorizontalChecker = 1;
	bool tempCorrect = false; // it means this picxel is not corecctly selected.
	for (int i = 0; i < numOfRows; i++) {
		for (int j = numberOfHorizontalChecker; j < numOfColumns - numberOfHorizontalChecker; j++) {
			if (input->at<Vec3b>(Point(j, i)) == Color) {
				tempCorrect = false;
				for (int k = 1; k <= numberOfHorizontalChecker; k++) {
					if (input->at<Vec3b>(Point(j + k, i)) == Color | input->at<Vec3b>(Point(j - k, i)) == Color) {
						tempCorrect = true;
						break;
					}
				}
				if (!tempCorrect) {
					input->at<Vec3b>(Point(j, i)) = background->at<Vec3b>(Point(j, i));
				}
			}
		}
	}


}


////////////////////////////////////////////////////////////////////
/// In this part we will detcte the stain.
////////////////////////////////////////////////////////////////////
void stainDetector(shared_ptr<Mat> background, shared_ptr<Mat> input, Vec3b Color) {
	for (int i = 0; i < numOfColumns; i++) {
		for (int j = 0; j < numOfRows; j++) {
			if (input->at<Vec3b>(Point(i, j)) == Color) {
				auto stain = make_shared<Stain> ();
				makeStain(background, input, stain, i, j, Color);
			}
		}
	}
}

void makeStain(shared_ptr<Mat> background, shared_ptr<Mat> input, shared_ptr<Stain> stain, int i, int j,Vec3b Color) {

	checkPoint(background, input,stain, i, j, Color);

}

void checkPoint(shared_ptr<Mat> background, shared_ptr<Mat> input,shared_ptr<Stain> stain, int i, int j, Vec3b Color) {
	if (input->at<Vec3b>(Point(i, j)) == Color) {
		checkPoint(background, input,stain, i + 1, j, Color);
		checkPoint(background, input,stain ,i, j + 1, Color);
		checkPoint(background, input, stain, i + 1, j + 1, Color);
		checkPoint(background, input, stain, i - 1, j, Color);
		checkPoint(background, input, stain, i, j - 1, Color);
		checkPoint(background, input, stain, i - 1, j - 1, Color);
	}
	if (input->at<Vec3b>(Point(i, j)) == Color) {
		input->at<Vec3b>(Point(i, j)) = background->at<Vec3b>(Point(i, j));
		stain->stainPoints.push_back(Point(i, j));
	}



}