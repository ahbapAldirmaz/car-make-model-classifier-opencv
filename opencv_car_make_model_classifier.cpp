// Copyright © 2019 by Spectrico
// Licensed under the MIT License

#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

template <typename T>
std::vector<size_t> SortIndexes(const std::vector<T> &v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });

	return idx;
}

std::vector<std::string> readClassNames(std::string filename)
{
	std::vector<std::string> classNames;

	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}

	fp.close();
	return classNames;
}

cv::Mat GetSquareImage(const cv::Mat& img, int target_width)
{
	int width = img.cols,
		height = img.rows;

	cv::Mat square = cv::Mat::zeros(target_width, target_width, img.type());

	int max_dim = (width >= height) ? width : height;
	double scale = ((double)target_width) / max_dim;
	cv::Rect roi;
	if (width >= height)
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = static_cast<int>(height * scale);
		roi.y = (target_width - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = static_cast<int>(width * scale);
		roi.x = (target_width - roi.width) / 2;
	}

	cv::resize(img, square(roi), roi.size());

	return square;
}

int main(int argc, char** argv)
{
	const std::string modelFile = "model-weights-spectrico-mmr-mobilenet-224x224-908A6A8C.pb";
	const int classifier_input_size = 224;

	cv::dnn::Net net;
	//! [Initialize network]
	net = cv::dnn::readNetFromTensorflow(modelFile);
	if (net.empty())
	{
		std::cerr << "Can't load network by using the model file: " << std::endl;
		std::cerr << modelFile << std::endl;
		exit(-1);
	}

	std::vector<std::string> classNames = readClassNames("labels.txt");

	std::string imageFile = argc == 2 ? argv[1] : "car.jpg";

	cv::Mat img = cv::imread(imageFile, cv::IMREAD_COLOR);
	if (img.empty() || !img.data)
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	img = GetSquareImage(img, classifier_input_size);

	//! [Prepare blob]
	cv::Mat inputBlob = cv::dnn::blobFromImage(img, 0.0078431372549019607843137254902, cv::Size(classifier_input_size, classifier_input_size), cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);   //Convert Mat to image batch
	std::string inBlobName = "input_1";
	//! [Set input blob]
	net.setInput(inputBlob, inBlobName);        //set the network input

	std::string outBlobName = "softmax/Softmax";

	cv::TickMeter tm;
	tm.start();

	cv::Mat result;
	//! [Make forward pass]
	result = net.forward(outBlobName);       //compute output

	tm.stop();

	std::cout << "Inference time, ms: " << tm.getTimeMilli() << std::endl;
	const int top_n = 3;
	std::cout << "Top " << std::to_string(top_n) << " probabilities: " << std::endl;
	cv::Mat probMat = result.reshape(1, 1);
	std::vector<float>vec(probMat.begin<float>(), probMat.end<float>());
	int top = 0;
	for (auto i : SortIndexes(vec))
	{
		std::string make_model = classNames.at(i);
		std::string make;
		std::string model;
		size_t pos = 0;
		pos = make_model.find("\t");
		if (pos != std::string::npos)
		{
			make = make_model.substr(0, pos);
			model = make_model.substr(pos + 1);
		}

		std::cout << "make: " << make << "\tmodel: " << model << "\tconfidence: " << vec[i] * 100 << " %" << std::endl;
		if (++top == top_n)
			break;
	}

	return 0;
}
