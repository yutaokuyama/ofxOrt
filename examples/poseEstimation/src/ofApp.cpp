#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	const ORTCHAR_T* modelName = L"pose.onnx";
	ort = new ofxOrt(modelName, true);

	ort->printModelInfo();
	img.load("sample.png");
	img.resize(256, 256);
	img.update();

	fbo.allocate(256, 256, GL_RGB);

	original.load("sample.png");

}

//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::draw() {
	fbo.begin();
	ofClear(0, 255);
	img.draw(0.0, 0.0, 256, 256);

	fbo.end();
	img.draw(0.0, 0.0, ofGetWidth() / 2.0, ofGetWidth() / 2.0);

	ofFloatPixels pix;
	fbo.getTexture().readToPixels(pix);


	pix.swapRgb();



	for (int i = 0; i < pix.getHeight(); i++) {
		for (int j = 0; j < pix.getWidth(); j++) {
			int index = i * pix.getHeight() + j;
			pix[3 * index + 0] = (pix[3 * index + 0] - 0.406) / 0.225;
			pix[3 * index + 1] = (pix[3 * index + 1] - 0.456) / 0.224;
			pix[3 * index + 2] = (pix[3 * index + 2] - 0.485) / 0.229;
		}
	}

	ofFloatPixels chw;
	ofxOrtUtils::hwc2chw(pix, chw);

	ofFloatImage img;
	img.setFromPixels(chw);
	img.update();


	inference(img);
	ofDrawBitmapString(ofToString(currentIndex), 0, 10);
	ofDrawBitmapString(ofToString(ofGetFrameRate()), 0, 20);

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	if (key == 'a') {
		currentIndex = (currentIndex + 1) % 16;
	}
}


void ofApp::inference(ofFloatImage& content) {

	const char* input_names[] = { "input" };
	const char* output_names[] = { "output" };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	ofFloatPixels pix;


	ofxOrtImageTensor input_tensor = ofxOrtImageTensor(memory_info, content.getTexture());
	ofxOrtImageTensor output_tensor = ofxOrtImageTensor(memory_info, 16, 64, 64, true);

	ort->forward(Ort::RunOptions{ nullptr }, input_names, &(input_tensor.getTensor()), 1, output_names, &(output_tensor.getTensor()), 1);



	std::vector<std::vector<float>> dstArray;
	ofxOrtUtils::splitImageDataArray(output_tensor.getTexData(), dstArray, 16, 64, 64);
	std::vector<ofFloatImage> heatmaps = ofxOrtUtils::buildImagesFromData(dstArray, 64, 64);

	heatmaps[currentIndex].draw(ofGetWidth() / 2.0, 0.0, ofGetWidth() / 2.0, ofGetWidth() / 2.0);


}
