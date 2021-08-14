#pragma once
#include "ofMain.h"
#include "onnxruntime_cxx_api.h"

class ofxOrtImageTensor {
public:
	ofxOrtImageTensor(Ort::MemoryInfo& memInfo, ofTexture& tex, bool isGrayscale = false) :tensor(nullptr)
	{
		ofFloatPixels pix;
		tex.readToPixels(pix);
		if (isGrayscale) {
			pix.setImageType(OF_IMAGE_GRAYSCALE);
		}

		texData = std::vector<float>{ pix.getData(), pix.getData() + pix.size() };
		data_shape = std::array<int64_t, 4>{ 1, int(pix.getNumChannels()), int(tex.getWidth()), int(tex.getHeight()) };
		tensor = Ort::Value::CreateTensor<float>(memInfo, texData.data(), texData.size(), data_shape.data(), data_shape.size());
	}

	ofxOrtImageTensor(Ort::MemoryInfo& memInfo, int numTex, int width, int height, bool isGrayscale = false) :tensor(nullptr)
	{

		ofFloatPixels tmp;
		if (isGrayscale) {
			tmp.allocate(width, height, OF_IMAGE_GRAYSCALE);
		}
		else {
			tmp.allocate(width, height, OF_IMAGE_COLOR);

		}
		texData.resize(tmp.size() * numTex);

		data_shape = std::array<int64_t, 4>{ 1, numTex, width, height };
		tensor = Ort::Value::CreateTensor<float>(memInfo, texData.data(), texData.size(), data_shape.data(), data_shape.size());
	}

	Ort::Value& getTensor() {
		return tensor;
	}

	std::vector<float>& getTexData() {
		return texData;
	};
private:
	Ort::Value tensor;
	std::vector<float> texData;
	std::array<int64_t, 4> data_shape;
};

class ofxOrt1DTensor {
public:
	ofxOrt1DTensor(Ort::MemoryInfo& memInfo, int numValue)
		:tensor_(nullptr)
	{
		data_.resize(numValue);
		data_shape_ = std::array<int64_t, 2>{ 1, int(data_.size()) };
		tensor_ = Ort::Value::CreateTensor<float>(memInfo, data_.data(), data_.size(), data_shape_.data(), data_shape_.size());
	}

	Ort::Value& getTensor() {
		return tensor_;
	}

	std::vector<float> getData() {
		return data_;
	}
private:
	std::vector<float> data_;
	Ort::Value tensor_;
	std::array<int64_t, 2> data_shape_;
};