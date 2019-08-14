#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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
};
int numOfColumns;
int numOfRows;
int thickness = 60;
int maxDisparity = 30;
int maxkernelSize = 35; // kernel size must be odd number.
typedef vector<pixel*> layerVector;
vector<layerVector> layers;
void ReadBothImages(shared_ptr<Mat>, shared_ptr<Mat>);
void Meshing(int, int, int, int, int);
double CalcDistance(int, int, int, int);
int CalcCost(shared_ptr<Mat>, shared_ptr<Mat>, int, int, int, int);
void stereo(shared_ptr<Mat>, shared_ptr<Mat>, layerVector*, int, int);
void makeResult(vector<layerVector>, int, int,int, string);

int main()
{

	auto rightImage = make_shared<Mat>();
	auto leftImage = make_shared<Mat>();
	ReadBothImages(leftImage, rightImage);
	numOfRows = leftImage->rows;
	numOfColumns = leftImage->cols;
	Meshing(numOfRows, numOfColumns, thickness, maxkernelSize, maxDisparity);
	for (int kernelSize = 3; kernelSize < maxkernelSize; kernelSize = kernelSize + 2) {
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
		string duration_s=to_string(value);
		
		makeResult(layers, numOfRows, numOfColumns, kernelSize, duration_s);
	}

	cout << layers.size() << endl;
	cout << "hello" << endl;
	int x;
	cin >> x;// Show our image inside it.
			 // Wait for a keystroke in the window
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
	int numOfLayers =int( CalcDistance(numOfRows, numOfColumns, 0, 0) / thickness);
	// the number 4 thai wrote there is for ensure that all of the image has suported... dont wworry... we have delete those who is null.
	for (int i = 1; i <= numOfLayers+4; i++) {
		layerVector tempLayer;
		layers.push_back(tempLayer);
	}
	for (int i = (kernelSize/2); i < numOfRows-(kernelSize/2) ; i++) {
		for (int j = (kernelSize / 2); j < numOfColumns-(kernelSize / 2)-maxDisparity; j++) {
			tempLayer=int(CalcDistance(numOfRows, numOfColumns, i, j)/ thickness);
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
	double tempDistance = sqrt(pow((row-numOfRows),2) + pow((column - int(numOfColumns / 2) + .05) , 2));
	return tempDistance;
}


////////////////////////////////////////////////////////////////////
/// In this part we clac disparity of each pixel.
////////////////////////////////////////////////////////////////////
void stereo(shared_ptr<Mat> leftImage, shared_ptr<Mat> rightImage, layerVector* layer, int kernelSize, int maxDisparity) {
	imshow("leftImage", *leftImage);
	imshow("rightImage", *rightImage);
	waitKey(12);
	int tempCost = 0;
	int tempDisparity = 0;
	for (int p = 1; p < layer->size(); p++) {
		//cout << "the alye size is " << layer->size() << endl;
		//cout << "disparity is  " << p << endl;
		double cost = 10000000;
	    tempCost = 0;
		tempDisparity = 0;
		for (int i = 0; i < maxDisparity; i++) {
			tempCost= CalcCost(leftImage, rightImage, (*layer)[p]->row, (*layer)[p]->column, kernelSize, i);
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
	int cost=0;
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
			cost = cost + int(pow((leftImage->at<uchar>(row + v, column + u) - (rightImage->at<uchar>(row + v , column + u+ disparity))),2));
		}
	}
	return cost;
}


////////////////////////////////////////////////////////////////////
/// In this part we can make the result.
////////////////////////////////////////////////////////////////////
void makeResult(vector<layerVector> layers, int numOfRows, int numOfColumns,int kernalSize, string Dutime) {
	Mat result(numOfRows, numOfColumns, CV_8UC1);
	for (int i = 0; i < layers.size(); i++) {
		for (int j = 0; j < layers[i].size(); j++) {
			result.at<uchar>(layers[i][j]->row, layers[i][j]->column) = uchar(255*layers[i][j]->disparity/30);
		}

	}
	string temp;
	
	temp = "result_KernelSize_" + to_string(kernalSize)+"_MaxDisparity_"+ to_string(maxDisparity)+"Time_"+Dutime +"s.png";
	imwrite(temp, result);
}