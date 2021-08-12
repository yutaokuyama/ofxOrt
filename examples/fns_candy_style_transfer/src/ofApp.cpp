#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	const ORTCHAR_T* modelName = L"candy.onnx";
	ort = new ofxOrt(modelName, true);

	fbo.allocate(720, 720,GL_RGB);
	fbo.begin();
	ofClear(0, 255);

	fbo.end();
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){


	fbo.begin();
	ofSetColor(ofRandom(0, 255), ofRandom(0, 255), ofRandom(0, 255));
	ofDrawCircle(mouseX, mouseY, ofRandom(10, 100));
	fbo.end();
	fbo.draw(0.0, 0.0);

	ofxOrtUtils::hwc_to_chw(fbo.getTexture(), pixCHW);
	ofFloatImage img_CHW;
	img_CHW.setFromPixels(pixCHW);
	img_CHW.update();
	inference(img_CHW);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if (key == ' ') {
		fbo.begin();
		ofClear(0, 255);

		fbo.end();
	}
}


void ofApp::inference(ofFloatImage content) {

	const char* input_names[] = { "inputImage" };
	const char* output_names[] = { "outputImage" };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	ofImage result("sample.jpg");
	result.resize(720, 720);
	result.update();

	ofFloatPixels pix;
	ofFloatPixels pix_result;

	content.getTexture().readToPixels(pix);
	content.getTexture().readToPixels(pix_result);


	ofxOrtImageTensor input_tensor = ofxOrtImageTensor(memory_info, content.getTexture());
	ofxOrtImageTensor output_tensor = ofxOrtImageTensor(memory_info, content.getTexture());


	ort->forward(Ort::RunOptions{ nullptr }, input_names, &(input_tensor.getTensor()), 1, output_names, &(output_tensor.getTensor()), 1);
	
	pix.setFromAlignedPixels(output_tensor.getTexData().data(), 720, 720, OF_PIXELS_RGB,720*3);

	ofxOrtUtils::chw_to_hwc(pix, pix_result, 720, 720);

	result.setFromPixels(pix_result);

	result.update();
	result.draw(720.0, 0.0);
}
