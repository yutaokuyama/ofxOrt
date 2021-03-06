#pragma once

#include "ofMain.h"
#include "ofxOrt.h"

class ofApp : public ofBaseApp {

public:
  void setup();
  void update();
  void draw();

  void buildModel();
  void inference();

  void clearFbos();
  void allocateFbos();

  void keyPressed(int key);
  void drawBins(std::vector<float> results);

  ofxOrt *ort;
  ofFbo screenFbo;
  ofFbo sampleFbo;
};
