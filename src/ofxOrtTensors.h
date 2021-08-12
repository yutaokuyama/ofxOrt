#pragma once
#include "ofMain.h"
#include "onnxruntime_cxx_api.h"

class ofxOrtImageTensor{
	public:
		ofxOrtImageTensor(Ort::MemoryInfo& memInfo, ofTexture tex, bool isGrayscale ) :tensor( nullptr )
		{
			ofFloatPixels pix;
			tex.readToPixels(pix);
			if(isGrayscale){
				pix.setImageType(OF_IMAGE_GRAYSCALE);
			}
			texData = std::vector<float>{pix.getData(), pix.getData() + pix.size() };
			data_shape = std::array<int64_t, 4>{ 1, int(pix.getNumChannels()), int(tex.getWidth()), int(tex.getHeight()) };
			tensor = Ort::Value::CreateTensor<float>(memInfo, texData.data(), texData.size(), data_shape.data(), data_shape.size());
		}
		Ort::Value& getTensor(){
			return tensor;
		}
	private:
		Ort::Value tensor;
		std::vector<float> texData;
		std::array<int64_t, 4> data_shape;
};