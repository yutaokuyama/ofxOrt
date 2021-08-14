#pragma once

#include "ofMain.h"
#include "ofxOrt.h"
#include "ofxCv.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void inference(ofFloatImage& content);



		ofxOrt* ort;
		ofFloatImage img;
		
		ofFbo fbo;

		int currentIndex = 0;
		ofImage original;
};
