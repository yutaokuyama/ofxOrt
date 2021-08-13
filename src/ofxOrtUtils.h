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

