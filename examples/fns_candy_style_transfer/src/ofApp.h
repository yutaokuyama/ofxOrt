#pragma once

#include "ofMain.h"
#include "ofxOrt.h"


class ThreadedInference : public ofThread {
public:

	void setup(const int width, const int height)
	{
		const ORTCHAR_T* modelName = L"candy.onnx";
		ort = new ofxOrt(modelName, true);
		ort->printModelInfo();
		_srcHWCPixels.allocate(width, height, OF_IMAGE_COLOR);
	}

	void threadedFunction() override {

		inference();
	}


	void inference() {

		if (_srcHWCPixels.getWidth() == 0) {
			abort();
		}
		ofFloatPixels pixCHW(_srcHWCPixels);
		ofxOrtUtils::rgb2chw(_srcHWCPixels, pixCHW, true, true, 1.0);

		const char* input_names[] = { "inputImage" };
		const char* output_names[] = { "outputImage" };

		auto memory_info =
			Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);


		ofxOrtImageTensor<float> input_tensor(memory_info, pixCHW, _srcHWCPixels.getWidth(), _srcHWCPixels.getHeight());
		ofxOrtImageTensor<float> output_tensor(memory_info, pixCHW, _srcHWCPixels.getWidth(), _srcHWCPixels.getHeight());

		ort->forward(Ort::RunOptions{ nullptr }, input_names,
			&(input_tensor.getTensor()), 1, output_names,
			&(output_tensor.getTensor()), 1);

		_dstPixels.setFromAlignedPixels(output_tensor.getTexData().data(),
			pixCHW.getWidth(), pixCHW.getHeight(),
			OF_PIXELS_RGB, pixCHW.getHeight() * 3);

		_dstPixels = (ofxOrtUtils::chw2hwc(_dstPixels, 1.0 / 255.0));
		_dstPixels.swapRgb();
		_isImageProcessed = true;


	}

	ofFloatPixels _srcHWCPixels;
	ofFloatPixels _dstPixels;
	bool _isImageProcessed = true;


private:
	ofxOrt* ort;

};


class ofApp : public ofBaseApp {

public:
	void setup() override;
	void update();
	void draw();
	void exit() override;
	void keyPressed(int key);
	void updateResultImage();

	void setImageToModel();
	void readImageFromModel();

	//ofFloatPixels inference(ofFloatPixels& content, int width, int height);


	ofFloatPixels pixCHW;

	ofFbo fbo;
	ofVideoGrabber grabber;
	ThreadedInference ort;
	ofImage resultImage;
	bool isFboReady = false;
	bool _isImageSet = false;

};
