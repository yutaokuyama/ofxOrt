#pragma once

#include "ofMain.h"
#include "ofxOrt.h"

class ThreadedInference: public ofThread {
public:
    void setup() {
        const ORTCHAR_T* modelName = L"candy.onnx";
        ort = new ofxOrt(modelName, true);
        ort->printModelInfo();
    }
    ofFloatPixels inference(ofFloatPixels& input,int width,int height) {
        const char* input_names[] = { "inputImage" };
        const char* output_names[] = { "outputImage" };

        auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);


        ofxOrtImageTensor<float> input_tensor(memory_info, input, width, height);
        ofxOrtImageTensor<float> output_tensor(memory_info, input, width, height);

        ort->forward(Ort::RunOptions{ nullptr }, input_names,
            &(input_tensor.getTensor()), 1, output_names,
            &(output_tensor.getTensor()), 1);

        ofFloatPixels pix_result;

        pix_result.setFromAlignedPixels(output_tensor.getTexData().data(),
            input.getWidth(), input.getHeight(),
            OF_PIXELS_RGB, input.getHeight() * 3);

        ofFloatPixels chwPixels(ofxOrtUtils::chw2hwc(pix_result, 1.0 / 255.0));
        chwPixels.swapRgb();
        return chwPixels;
    }

private:
    ofxOrt* ort;

};


class ofApp : public ofBaseApp {

public:
  void setup();
  void update();
  void draw();
  void keyPressed(int key);

  ofFloatPixels inference(ofFloatPixels& content, int width, int height);


  ofFloatPixels pixCHW;

  ofFbo fbo;
  ofVideoGrabber grabber;
  ofxOrt* ort;
};
