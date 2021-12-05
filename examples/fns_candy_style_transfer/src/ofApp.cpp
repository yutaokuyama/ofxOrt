#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	const ORTCHAR_T* modelName = L"candy.onnx";

	fbo.allocate(720, 720, GL_RGB);
	fbo.begin();
	ofClear(0, 255);
	fbo.end();


	ort.setup(720, 720);

}

//--------------------------------------------------------------
void ofApp::update() {

	fbo.begin();
	ofSetColor(ofRandom(0, 255), ofRandom(0, 255), ofRandom(0, 255));
	ofDrawCircle(mouseX, mouseY, ofRandom(10, 100));
	fbo.end();


	//I'm not sure if this is the right implementation for exclusions.
	if (fbo.checkStatus()) {


		ort.lock();
		if (!_isImageSet && !ort.isThreadRunning()) {
			setImageToModel();

		}
		if (ort._isImageProcessed && _isImageSet) {
			readImageFromModel();
		}

		ort.unlock();



		if (!ort.isThreadRunning()) {
			ort.startThread(false);
		}



	}

}

//--------------------------------------------------------------
void ofApp::draw() {

	fbo.draw(0.0, 0.0);
	resultImage.draw(720.0, 0.0);
	ofDrawBitmapString(ofToString(ofGetFrameRate()), 0, 10);

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	if (key == ' ') {
		fbo.begin();
		ofClear(0, 255);
		fbo.end();
	}
}

void ofApp::updateResultImage() {
	resultImage.setFromPixels(ort._dstPixels);
	resultImage.update();
}

void ofApp::exit() {


	ort.stopThread();



}

void ofApp::setImageToModel() {

	fbo.getTexture().readToPixels(ort._srcHWCPixels);


	_isImageSet = true;
	ort._isImageProcessed = false;

}

void ofApp::readImageFromModel() {

	updateResultImage();
	_isImageSet = false;
	ort._isImageProcessed = false;
}