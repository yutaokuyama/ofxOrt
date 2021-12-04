#pragma once

#include "ofMain.h"
#include "ofxOrt.h"
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
