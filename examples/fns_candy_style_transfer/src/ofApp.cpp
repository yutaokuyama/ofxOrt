#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	fbo.allocate(720, 720, GL_RGB);
	ort.setup(720, 720);
}

//--------------------------------------------------------------
void ofApp::update() {
	if (isMousePressed) {
		fbo.begin();
		ofSetColor(ofRandom(0, 255), ofRandom(0, 255), ofRandom(0, 255));
		ofDrawCircle(ofGetMouseX(), ofGetMouseY(), ofRandom(10, 100));
		fbo.end();
	}
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

//--------------------------------------------------------------
void ofApp::draw() {
	fbo.draw(0.0, 0.0);
	if (resultImage.isAllocated()) {
		resultImage.draw(720.0, 0.0);
	}
	ofDrawBitmapString(ofToString(ofGetFrameRate()), 0, 10);
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {
	fbo.begin();
	ofSetColor(ofRandom(0, 255), ofRandom(0, 255), ofRandom(0, 255));
	ofDrawCircle(ofGetMouseX(), ofGetMouseY(), ofRandom(10, 100));
	fbo.end();
	isMousePressed = true;
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {
	isMousePressed = false;
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
	resultImage.loadData(ort._dstPixels);
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
