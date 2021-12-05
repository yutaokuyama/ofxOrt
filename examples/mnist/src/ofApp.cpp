#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
  allocateFbos();
  clearFbos();
  buildModel();
  ort->printModelInfo();
}

void ofApp::buildModel() {
  const ORTCHAR_T *modelName = L"mnist.onnx";
  ort = new ofxOrt(modelName, true);
}

void ofApp::inference() {

  const char *input_names[] = {"Input3"};
  const char *output_names[] = {"Plus214_Output_0"};

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  ofFloatPixels pix;
  sampleFbo.getTexture().readToPixels(pix);

  ofxOrtImageTensor<float> input_tensor_original(memory_info,
                                                 pix,pix.getWidth(),pix.getHeight(), true);
  ofxOrt1DTensor<float> output_tensor_original(memory_info, 10);

  ort->forward(Ort::RunOptions{nullptr}, input_names,
               &(input_tensor_original.getTensor()), 1, output_names,
               &(output_tensor_original.getTensor()), 1);
  drawBins(output_tensor_original.getData());
}

//--------------------------------------------------------------
void ofApp::drawBins(std::vector<float> results) {
  float min = 9999;
  float max = -9999;
  float width = 100;
  float lineHeight = 15;
  int max_index = 0;
  for (int i = 0; i < results.size(); i++) {
    if (min > results[i]) {
      min = results[i];
    }
    if (max < results[i]) {
      max = results[i];
      max_index = i;
    }
  }

  for (int i = 0; i < results.size(); i++) {

    float range = max - min;
    float normalizedValue = (results[i] + abs(min)) / range;
    ofDrawBitmapString(ofToString(i), 20, lineHeight * i + 10);
    ofDrawRectangle(50, lineHeight * i, normalizedValue * width, lineHeight);
  }
  ofDrawBitmapString("looks like " + ofToString(max_index), 10,
                     lineHeight * 10 + 10);
  ofDrawBitmapString("FPS: " + ofToString(ofGetFrameRate()), 10,
                     lineHeight * 11 + 10);
  ofDrawBitmapString("Clear canvas: c", 10, lineHeight * 12 + 10);
}

void ofApp::update() {}

//--------------------------------------------------------------
void ofApp::draw() {
  screenFbo.begin();
  if (ofGetMousePressed()) {
    ofDrawCircle(mouseX, mouseY, 28);
  }
  screenFbo.end();
  screenFbo.draw(0.0, 0.0);

  sampleFbo.begin();
  ofClear(0, 255);
  screenFbo.draw(0, 0, 28, 28);
  sampleFbo.end();

  inference();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
  if (key == ' ') {
    clearFbos();
  }
}

void ofApp::clearFbos() {
  screenFbo.begin();
  ofClear(0, 255);
  screenFbo.end();

  sampleFbo.begin();
  ofClear(0, 255);
  sampleFbo.end();
}

void ofApp::allocateFbos() {
  screenFbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGB);
  sampleFbo.allocate(28, 28, GL_RGB);
}