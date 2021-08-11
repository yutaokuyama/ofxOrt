#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	allocateFbos();
	clearFbos();
	buildModel();

}

void ofApp::buildModel() {
	const ORTCHAR_T* modelName = L"mnist.onnx";
	ort = new ofxOrt(modelName, true);
}

void ofApp::inference() {

	static constexpr const int img_w = 28;
	static constexpr const int img_h = 28;

	std::array<float, img_w* img_h> input_image_{};
	ofPixels pix;
	sampleFbo.getTexture().readToPixels(pix);
	for (int i = 0; i < pix.size() / 3; i++) {
		float value = (pix.getData()[i * 3] > 1) ? 1 : 0;
		input_image_[i] = value;
	}

	const char* input_names[] = { "Input3" };
	const char* output_names[] = { "Plus214_Output_0" };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	std::array<float, 10> results_{};
	int64_t result_{ 0 };

	Ort::Value input_tensor{ nullptr };
	std::array<int64_t, 4> input_shape{ 1, 1, img_w, img_h };
	Ort::Value output_tensor{ nullptr };
	std::array<int64_t, 2> output_shape{ 1, 10 };


	input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape.data(), input_shape.size());
	output_tensor = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape.data(), output_shape.size());

	ort->forward(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, &output_tensor, 1);

	drawBins(results_);
}


//--------------------------------------------------------------
void ofApp::drawBins(std::array<float, 10> results) {
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
		ofDrawRectangle(50,lineHeight*i, normalizedValue * width, lineHeight);
	}
	ofDrawBitmapString("looks like "+ofToString(max_index), 10, lineHeight * 10 + 10);
	ofDrawBitmapString("FPS: " + ofToString(ofGetFrameRate()), 10, lineHeight * 11 + 10);
	ofDrawBitmapString("Clear canvas: c", 10, lineHeight * 12 + 10);
}

void ofApp::update() {

}

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