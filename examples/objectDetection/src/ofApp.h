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


	void objectDetectionWithModelOutput(const std::vector<float>& inferenceOutput,
		std::vector<std::array<float, 4>>& bboxes,
		std::vector<float>& scores,
		std::vector<uint64_t>& classIndices, const float confidenceThresh);

	void drawBoundingBox(std::array<float, 4>& bbox, std::string& className, float score);

	ofxOrt* ort;
	ofFbo fbo;

	ofFloatImage original;

	const int IMG_WIDTH = 416;
	const int IMG_HEIGHT = 416;
	const int IMG_CHANNEL = 3;
	const int FEATURE_MAP_SIZE = 13 * 13;
	const int NUM_BOXES = 1 * 13 * 13 * 125;
	const int NUM_ANCHORS = 5;
	const float ANCHORS[10] = { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };

	float scale = 1.0;

};
