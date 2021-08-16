#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxOrt.h"

class ofApp : public ofBaseApp {

public:
  void setup();
  void update();
  void draw();

  void inference(ofFloatImage &content);

  ofxOrt *ort;
  ofFbo fbo;

  ofVideoPlayer vid;
};
