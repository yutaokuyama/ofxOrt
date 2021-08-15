#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ort = new ofxOrt(ORT_TSTR("tinyyolov2-8.onnx"), true);
	ort->printModelInfo();

	original.load("cat.jpg");

	fbo.allocate(416, 416, GL_RGB);
}

//--------------------------------------------------------------
void ofApp::update() {
	scale = (ofGetWidth() / 1.3) / original.getWidth();
}

//--------------------------------------------------------------
void ofApp::draw() {
	fbo.begin();
	ofClear(0, 255);
	original.draw(0.0, 0.0, fbo.getWidth(), fbo.getHeight());
	fbo.end();

	original.draw(0.0, 0.0, original.getWidth() * scale, original.getHeight() * scale);

	ofFloatPixels pix;
	ofFloatPixels chw;
	fbo.getTexture().readToPixels(pix);

	ofxOrtUtils::rgb2chw(pix, chw, false, true, 255.0);
	ofFloatImage src;
	src.setFromPixels(chw);
	src.update();

	inference(src);

}

//--------------------------------------------------------------
void ofApp::inference(ofFloatImage& content) {

	const char* input_names[] = { "image" };
	const char* output_names[] = { "grid" };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	const int numChannels = 125;
	const int outWidth = 13;
	const int outHeight = 13;

	ofxOrtImageTensor<float> input_tensor(memory_info, content.getTexture());
	ofxOrtImageTensor<float> output_tensor(memory_info, numChannels, outWidth, outHeight, true);
	ort->forward(Ort::RunOptions{ nullptr }, input_names, &(input_tensor.getTensor()), 1, output_names, &(output_tensor.getTensor()), 1);

	std::vector<std::array<float, 4>> bboxes;
	std::vector<float> scores;
	std::vector<uint64_t> classIndices;
	const float confidenceThresh = 0.5;

	objectDetectionWithModelOutput(output_tensor.getTexData(), bboxes, scores, classIndices, confidenceThresh);

	std::vector<string> labels{
		"aeroplane", "bicycle", "bird", "boat", "bottle",
		"bus", "car", "cat", "chair", "cow",
		"diningtable", "dog", "horse", "motorbike", "person",
		"pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	for (int i = 0; i < bboxes.size(); i++) {
		drawBoundingBox(bboxes[i], labels[classIndices[i]], scores[i]);

	}
}

void ofApp::exit() {
	delete ort;
}

void ofApp::objectDetectionWithModelOutput(const std::vector<float>& inferenceOutput,
	std::vector<std::array<float, 4>>& bboxes,
	std::vector<float>& scores,
	std::vector<uint64_t>& classIndices, const float confidenceThresh)
{

	std::vector<float> outputData{ inferenceOutput.begin(), inferenceOutput.begin() + NUM_BOXES };

	const int m_numClasses = 20;
	float tmpScores[m_numClasses];



	for (uint64_t i = 0; i < FEATURE_MAP_SIZE; ++i) {
		for (uint64_t j = 0; j < NUM_ANCHORS; ++j) {
			for (uint64_t k = 0; k < m_numClasses; ++k) {
				tmpScores[k] = outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + k + 5)];
			}

			ofxOrtUtils::softmax(tmpScores, m_numClasses);
			uint64_t maxIdx = std::distance(tmpScores, std::max_element(tmpScores, tmpScores + m_numClasses));
			float probability = tmpScores[maxIdx];

			if (ofxOrtUtils::sigmoid(outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + 4)]) * probability >=
				confidenceThresh) {
				float xcenter =
					(ofxOrtUtils::sigmoid(outputData[i + FEATURE_MAP_SIZE * (m_numClasses + 5 * j)]) + i % 13) * 32.0;
				float ycenter =
					(ofxOrtUtils::sigmoid(outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + 1)]) + i % 13) * 32.0;

				float width =
					expf(outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + 2)]) * ANCHORS[2 * j] * 32.0;
				float height =
					expf(outputData[i + FEATURE_MAP_SIZE * ((m_numClasses + 5) * j + 3)]) * ANCHORS[2 * j + 1] * 32.0;

				float xmin = std::max<float>(xcenter - width / 2, 0.0);
				float ymin = std::max<float>(ycenter - height / 2, 0.0);
				float xmax = std::min<float>(xcenter + width / 2, IMG_WIDTH);
				float ymax = std::min<float>(ycenter + height / 2, IMG_HEIGHT);
				bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
				scores.emplace_back(probability);
				classIndices.emplace_back(maxIdx);
			}
		}
	}
}

void ofApp::drawBoundingBox(std::array<float, 4>& bbox, std::string& className, float score) {

	ofPushMatrix();
	ofScale((original.getWidth() / fbo.getWidth()) * scale, (original.getHeight() / fbo.getHeight()) * scale);
	ofTranslate(bbox[0], bbox[1]);
	ofFill();
	ofSetColor(255);
	ofDrawRectangle(0.0, 0.0, abs(bbox[0] - bbox[2]), 14.0);
	ofSetColor(0);
	ofDrawBitmapString(ofToString(className) + " " + ofToString(score), 0.0, 12.0);
	ofSetColor(255);
	ofNoFill();
	ofDrawRectangle(0, 0.0, abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3]));
	ofFill();
	ofPopMatrix();
}