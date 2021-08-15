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

	static void rgb2chw(ofFloatPixels src, ofFloatPixels& dst, bool shouldNormalize, bool shouldSwapRg) {
		//TODO::extremely inefficient
		if (shouldSwapRg) {
			src.swapRgb();
		}
		if (shouldNormalize) {
			normalizePixel(src);
		}
		hwc2chw(src, dst);
	}


	static void rgb2chw(ofFloatPixels src, ofFloatPixels& dst, bool shouldNormalize, bool shouldSwapRg, const float scaleValue = 1.0) {
		//TODO::extremely inefficient
		if (shouldSwapRg) {
			src.swapRgb();
		}
		if (shouldNormalize) {
			normalizePixel(src);
		}
		scalePixelValue(src, scaleValue);

		hwc2chw(src, dst);
	}



	static void scalePixelValue(ofFloatPixels& pixels, const float value) {
		for (int i = 0; i < pixels.getHeight(); i++) {
			for (int j = 0; j < pixels.getWidth(); j++) {
				int index = i * pixels.getHeight() + j;
				pixels[3 * index + 0] *= value;
				pixels[3 * index + 1] *= value;
				pixels[3 * index + 2] *= value;
			}
		}
	}

	static void normalizePixel(ofFloatPixels& pixels) {
		for (int i = 0; i < pixels.getHeight(); i++) {
			for (int j = 0; j < pixels.getWidth(); j++) {
				int index = i * pixels.getHeight() + j;
				pixels[3 * index + 0] = (pixels[3 * index + 0] - 0.406) / 0.225;
				pixels[3 * index + 1] = (pixels[3 * index + 1] - 0.456) / 0.224;
				pixels[3 * index + 2] = (pixels[3 * index + 2] - 0.485) / 0.229;
			}
		}
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

	static void ofxOrtUtils::softmax(float* input, const size_t inputLen)
	{
		const float maxVal = *std::max_element(input, input + inputLen);

		const float sum = std::accumulate(input, input + inputLen, 0.0,
			[&](float a, const float b) { return std::move(a) + expf(b - maxVal); });

		const float offset = maxVal + logf(sum);
		for (auto it = input; it != (input + inputLen); ++it) {
			*it = expf(*it - offset);
		}
	}

	static float ofxOrtUtils::sigmoid(const float x)
	{
		return 1.0 / (1.0 + expf(-x));
	}

	static std::string getONNXTensorElementDataTypeName(ONNXTensorElementDataType id) {
		switch (id) {
		case 0:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED");
		case 1:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT");   // maps to c type float
		case 2:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8");   // maps to c type uint8_
		case 3:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8");    // maps to c type 
		case 4:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16");  // maps to c type uint16_t

		case 5:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16");   // maps to c type int16_t

		case 6:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32");   // maps to c type int32_t

		case 7:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64");   // maps to c type int64_t

		case 8:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING");  // maps to c++ type std::string

		case 9:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL");

		case 10:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16");

		case 11:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE");      // maps to c type double
		case 12:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32");      // maps to c type uint32_t

		case 13:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64");      // maps to c type uint64_t

		case 14:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64");   // complex with float32 real and imaginary components

		case 15:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128");  // complex with float64 real and imaginary components

		case 16:
			return std::string("ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16");     // Non-IEEE floating-point format based on IEEE754 single-precision

		}
	}

};