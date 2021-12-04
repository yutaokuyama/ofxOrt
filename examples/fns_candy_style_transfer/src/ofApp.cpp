#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
  const ORTCHAR_T *modelName = L"candy.onnx";
  ort = new ofxOrt(modelName, true);

  fbo.allocate(720, 720, GL_RGB);
  fbo.begin();
  ofClear(0, 255);
  fbo.end();

  ort->printModelInfo();
}

//--------------------------------------------------------------
void ofApp::update() {}

//--------------------------------------------------------------
void ofApp::draw() {

  fbo.begin();
  ofSetColor(ofRandom(0, 255), ofRandom(0, 255), ofRandom(0, 255));
  ofDrawCircle(mouseX, mouseY, ofRandom(10, 100));
  fbo.end();
  fbo.draw(0.0, 0.0);

  ofFloatPixels hwcPixels;
  fbo.getTexture().readToPixels(hwcPixels);
  ofxOrtUtils::rgb2chw(hwcPixels, pixCHW, true, true, 1.0);
  ofFloatImage img_CHW;
  img_CHW.setFromPixels(pixCHW);
  img_CHW.update();
  inference(img_CHW);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
  if (key == ' ') {
    fbo.begin();
    ofClear(0, 255);
    fbo.end();
  }
}

ofFloatPixels ofApp::inference(ofFloatPixels &content,int width,int height) {

  const char *input_names[] = {"inputImage"};
  const char *output_names[] = {"outputImage"};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);


  ofxOrtImageTensor<float> input_tensor(memory_info, content, width,height);
  ofxOrtImageTensor<float> output_tensor(memory_info, content, width, height);

  ort->forward(Ort::RunOptions{nullptr}, input_names,
               &(input_tensor.getTensor()), 1, output_names,
               &(output_tensor.getTensor()), 1);

  ofFloatPixels pix_result;

  pix_result.setFromAlignedPixels(output_tensor.getTexData().data(),
                           content.getWidth(), content.getHeight(),
                           OF_PIXELS_RGB, content.getHeight() * 3);

  ofFloatPixels chwPixels(ofxOrtUtils::chw2hwc(pix_result, 1.0 / 255.0));
  chwPixels.swapRgb();
  return chwPixels;

}
