#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	const ORTCHAR_T* modelName = L"pose.onnx";
	ort = new ofxOrt(modelName, true);

	ort->printModelInfo();

	vid.load("danceInOcean.mp4");
	vid.play();
	
	fbo.allocate(256, 256, GL_RGB);
}

//--------------------------------------------------------------
void ofApp::update() {
	vid.update();
}

//--------------------------------------------------------------
void ofApp::draw() {
	fbo.begin();
	ofClear(0, 255);
	vid.draw(0.0, 0.0, 256, 256);
	fbo.end();

	vid.draw(0.0, 0.0, ofGetWidth() / 2.0, ofGetWidth() / 2.0);

	ofFloatPixels pix;
	ofFloatPixels chw;
	fbo.getTexture().readToPixels(pix);

	ofxOrtUtils::rgb2chw(pix, chw, true, true);

	ofFloatImage src;
	src.setFromPixels(chw);
	src.update();

	inference(src);

}

//--------------------------------------------------------------
void ofApp::inference(ofFloatImage& content) {

	const char* input_names[] = { "input" };
	const char* output_names[] = { "output" };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	const int numChannel = 16;
	const int width = 64;
	const int height = 64;

	ofFloatPixels pix;

	ofxOrtImageTensor<float> input_tensor(memory_info, content.getTexture());
	ofxOrtImageTensor<float> output_tensor(memory_info, numChannel, width, height, true);

	ort->forward(Ort::RunOptions{ nullptr }, input_names, &(input_tensor.getTensor()), 1, output_names, &(output_tensor.getTensor()), 1);

	std::vector<std::vector<float>> dstArray;
	ofxOrtUtils::splitImageDataArray(output_tensor.getTexData(), dstArray, numChannel, width, height);
	std::vector<ofFloatImage> heatmaps = ofxOrtUtils::buildImagesFromData(dstArray, width, height);

	for (int i = 0; i < numChannel; i++) {
		int size = (ofGetWidth() / 2.0) / 4;
		glm::vec2 offset(size * (i / 4), size * (i % 4));

		//draw heatmaps
		heatmaps[i].draw(ofGetWidth() / 2.0 + offset.x, offset.y, size, size);


		//draw joint on original image
		cv::Point jointPosition;
		cv::Mat heatmatMat= ofxCv::toCv(heatmaps[i]);
		cv::minMaxLoc(heatmatMat, NULL, NULL, NULL, &jointPosition);
		glm::vec2 jointPointAsOf = ofxCv::toOf(jointPosition);
		float scale = (ofGetWidth() / 2.0) / width;
		ofDrawCircle(jointPointAsOf.x * scale, jointPointAsOf.y * scale, 2);
	}

}
