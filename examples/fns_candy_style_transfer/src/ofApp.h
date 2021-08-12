#pragma once

#include "ofMain.h"
#include "ofxOrt.h"
class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
		void keyPressed(int key);

		void inference(ofFloatImage content);
		ofxOrt* ort;

		ofImage img;
		
		ofFloatPixels pixCHW;
		glm::vec2 originalSize;

		ofFbo fbo;
		ofVideoGrabber grabber;
};
