#pragma once

#include "ofMain.h"
#include "ofxOrt.h"
#include "ofxCv.h"

class ofApp : public ofBaseApp {

public:
	void setup();
	void update();
	void draw();
	void exit();

	void inference(ofFloatImage& content);


	void ofApp::postProcess(const std::vector<float>& inferenceOutput,
		std::vector<std::array<float, 4>>& bboxes,
		std::vector<float>& scores,
		std::vector<uint64_t>& classIndices, const float confidenceThresh);

	ofxOrt* ort;
	ofFbo fbo;

	ofVideoPlayer vid;
	ofVideoGrabber grabber;
	ofFloatImage original;



};
