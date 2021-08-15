#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	const ORTCHAR_T* modelName = L"tinyyolov2-8.onnx";
	ort = new ofxOrt(modelName, true);

	ort->printModelInfo();

	//vid.load("danceInOcean.mp4");
	//vid.play();

	original.load("image2.jpg");
	original.setImageType(OF_IMAGE_COLOR);
	original.resize(416, 416);
	original.update();
	fbo.allocate(416, 416, GL_RGB);


	grabber.setDeviceID(1);
	grabber.initGrabber(416, 416);

}

//--------------------------------------------------------------
void ofApp::update() {
	grabber.update();
}

//--------------------------------------------------------------
void ofApp::draw() {
	fbo.begin();
	ofClear(0, 255);
	original.draw(0.0, 0.0, fbo.getWidth(), fbo.getHeight());
	fbo.end();

	original.draw(0.0, 0.0, 416, 416);

	ofFloatPixels pix;
	ofFloatPixels chw;
	fbo.getTexture().readToPixels(pix);

	ofxOrtUtils::rgb2chw(pix, chw, false, true,255.0);
	ofFloatImage src;
	src.setFromPixels(chw);
	src.update();


	//vid.
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

	postProcess(output_tensor.getTexData(), bboxes, scores, classIndices, confidenceThresh);

	std::vector<string> labels{
		"aeroplane", "bicycle", "bird", "boat", "bottle",
		"bus", "car", "cat", "chair", "cow",
		"diningtable", "dog", "horse", "motorbike", "person",
		"pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	for (int i = 0; i < bboxes.size(); i++) {

		ofNoFill();
		std::cout << "---------------" << std::endl;
		std::cout << "class:" << labels[int(classIndices[i])] << std::endl;
		std::cout << "xmin:" << bboxes[i][0] << " ymin: " << bboxes[i][1] << " xmax : " << bboxes[i][2] << " ymax : " << bboxes[i][3] << std::endl;
		ofDrawRectangle(bboxes[i][0], bboxes[i][1], abs(bboxes[i][0] - bboxes[i][2]), abs(bboxes[i][1] - bboxes[i][3]));
		ofFill();
	}
	ofFloatPixels pix;

	std::vector<std::vector<float>> dstArray;
	ofxOrtUtils::splitImageDataArray(output_tensor.getTexData(), dstArray, numChannels, outWidth, outHeight);
	std::vector<ofFloatImage> featureMaps = ofxOrtUtils::buildImagesFromData(dstArray, outWidth, outHeight);

	for (int i = 0; i < numChannels; i++) {
		int size = (ofGetWidth() / 2.0) / 12;
		glm::vec2 offset(size * (i / 12), size * (i % 12));

		//draw heatmaps
		featureMaps[i].draw(ofGetWidth() / 2.0 + offset.x, offset.y, size, size);
	}

}

void ofApp::exit() {
	delete ort;
}



void ofApp::postProcess(const std::vector<float>& inferenceOutput,
	std::vector<std::array<float, 4>>& bboxes,
	std::vector<float>& scores,
	std::vector<uint64_t>& classIndices, const float confidenceThresh)
{

	const int IMG_WIDTH = 416;
	const int IMG_HEIGHT = 416;
	const int IMG_CHANNEL = 3;
	const int FEATURE_MAP_SIZE = 13 * 13;
	const int NUM_BOXES = 1 * 13 * 13 * 125;
	const int NUM_ANCHORS = 5;
	const float ANCHORS[10] = { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };

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