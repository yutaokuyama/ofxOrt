#pragma once
#include "ofMain.h"
#include "onnxruntime_cxx_api.h"
#include "ofxCv.h"

class ofxOrtUtils {
public:
	static void hwc2chw(ofFloatPixels& hwc, ofFloatPixels& chw) {

		cv::Mat hwcMat = ofxCv::toCv(hwc);
		cv::Mat chwMat;
		cv::dnn::blobFromImage(hwcMat, chwMat);
		const float* data = chwMat.ptr<float>(0);

		chw.setFromAlignedPixels(data, hwc.getWidth(), hwc.getHeight(), OF_PIXELS_RGB, hwc.getHeight() * hwc.getNumChannels());
	}
	//TODO::extremely inefficient

	static void chw2hwc(const ofFloatPixels& pixels_chw, ofFloatPixels& pixels_hwc, bool normalize = false) {

		int stride = int(pixels_chw.getWidth() * pixels_chw.getHeight());

		for (int c = 0; c != 3; ++c) {
			int t = c * stride;
			for (int i = 0; i != stride; ++i) {
				float f = pixels_chw[t + i];
				if (f < 0.f || f > 255.0f) f = 0;
				pixels_hwc[i * 3 + c] = f;
				if (normalize) {
					pixels_hwc[i * 3 + c] /= 255.0;
				}
			}
		}
	}
};

	static void splitImageDataArray(const std::vector<float> data, std::vector<std::vector<float>>& dstArray, const int numTex, const int width, const int height) {
		//TODO: grayscale only
		//dstArray.resize(numTex);
		const int offset = (width * height);
		for (int i = 0; i < numTex; i++) {
			dstArray.emplace_back(std::vector<float>({ data.begin() + offset * i, data.begin() + offset * i + offset }));
		}
	}

	static std::vector<ofFloatImage> buildImagesFromData(const std::vector<std::vector<float>> data, const int width, const int height) {
		std::vector<ofFloatImage> images;
		//TODO: grayscale only
		for (auto& value : data) {
			ofFloatPixels pixels;
			ofFloatImage img;
			pixels.setFromAlignedPixels(value.data(), width, height, OF_PIXELS_GRAY, height);
			img.setFromPixels(pixels);
			img.update();
			images.push_back(img);
		}
		return images;
	}
};