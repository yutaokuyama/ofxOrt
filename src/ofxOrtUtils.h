#pragma once
#include "ofMain.h"
#include "onnxruntime_cxx_api.h"

class ofxOrtUtils {
public:
	//TODO::extremely inefficient
	static void hwc_to_chw(const ofTexture& tex, ofFloatPixels& pixels_chw, bool normalize = false) {

		ofFloatPixels pixels_hwc;
		pixels_chw.allocate(tex.getWidth(), tex.getHeight(), OF_IMAGE_COLOR);
		tex.readToPixels(pixels_hwc);
		int stride = int(int(tex.getHeight()) * int(tex.getWidth()));

		for (int i = 0; i != stride; ++i) {
			for (int c = 0; c != 3; ++c) {
				pixels_chw[c * stride + i] = pixels_hwc[i * 3 + c];
				if (normalize) {
					pixels_hwc[i * 3 + c] /= 255.0;
				}
			}
		}
	}

	static void hwc_to_chw(const ofFloatPixels& pixels_hwc, ofFloatPixels& pixels_chw, int width, int height, bool normalize = false) {

		int stride = width * height;
		for (int i = 0; i != stride; ++i) {
			for (int c = 0; c != 3; ++c) {
				pixels_chw[c * stride + i] = pixels_hwc[i * 3 + c];
				if (normalize) {
					pixels_chw[c * stride + i] /= 255.0;
				}
			}
		}
	}

	static void chw_to_hwc(const ofTexture& tex, ofFloatPixels& pixels_hwc, bool normalize = false) {

		ofFloatPixels pixels_chw;
		tex.readToPixels(pixels_hwc);
		tex.readToPixels(pixels_chw);
		int stride = int(tex.getHeight() * tex.getWidth());

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
	static void chw_to_hwc(const ofFloatPixels& pixels_chw, ofFloatPixels& pixels_hwc, int width, int height, bool normalize = false) {

		int stride = int(width * height);

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

