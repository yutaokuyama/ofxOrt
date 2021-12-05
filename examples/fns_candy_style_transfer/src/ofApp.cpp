#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
  const ORTCHAR_T *modelName = L"candy.onnx";

  fbo.allocate(720, 720, GL_RGB);
  fbo.begin();
  ofClear(0, 255);
  fbo.end();


  ort.setup(720, 720);
  grabber.setDeviceID(1);
  grabber.initGrabber(600,400);
  
}

//--------------------------------------------------------------
void ofApp::update() {
    grabber.update();
    if (grabber.isFrameNew()) {
        fbo.begin();
        ofClear(0, 255);
        //ofSetColor(ofRandom(0, 255), ofRandom(0, 255), ofRandom(0, 255));
        //ofDrawCircle(mouseX, mouseY, ofRandom(10, 100));
        grabber.draw(0.0, 0.0, fbo.getWidth(), fbo.getHeight());
        fbo.end();
        isFboReady = true;
    }
    ort.lock();
    if (!ort._isImageSeted) {
        setImageToModel();
    }
    if (ort._isImageProcessed && ort._isImageSeted) {
        readImageFromModel();
    }

    ort.unlock();
    if (!ort.isThreadRunning() && ort._isImageSeted && !ort._isImageProcessed) {
        ort.startThread(true);
    }

}

//--------------------------------------------------------------
void ofApp::draw() {

  fbo.draw(0.0, 0.0);
  resultImage.draw(720.0, 0.0);
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
    ort._isImageSeted = true;
    ort._isImageProcessed = false;

}

void ofApp::readImageFromModel() {

    updateResultImage();
    ort._isImageSeted = false;
    ort._isImageProcessed = false;
}