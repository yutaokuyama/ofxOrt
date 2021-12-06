#pragma once

#include "ofMain.h"
#include "ofxOrtTensors.h"
#include "ofxOrtUtils.h"
#include "onnxruntime_cxx_api.h"

class ofxOrt {
public:
	ofxOrt(const ORTCHAR_T* modelName, bool useCUDA);
	Ort::Session* getSession();

	void forward(const Ort::RunOptions& run_options,
		const char* const* input_names, const Ort::Value* input_values,
		size_t input_count, const char* const* output_names,
		Ort::Value* output_values, size_t output_count);

	std::vector<Ort::Value> forward(const Ort::RunOptions& run_options,
		const char* const* input_names, const Ort::Value* input_values,
		size_t input_count, const char* const* output_names,
		size_t output_count);



	size_t getSessionInputCount() const;
	size_t getSessionOutputCount() const;
	std::vector<std::string>  getSessionInputNames(size_t intputCount) const;
	std::vector<std::string>  getSessionOutputNames(size_t outputCount) const;

	std::vector<ONNXTensorElementDataType> getSessionInputTypes(size_t intputCount) const;
	std::vector<ONNXTensorElementDataType> getSessionOutputTypes(size_t outputCount) const;

	std::vector<std::vector<int64_t>> getInputDims(size_t intputCount) const;
	std::vector<std::vector<int64_t>>  getOutputDims(size_t outputCount) const;

	void printModelInfo() const;

private:
	void enableCUDA(Ort::SessionOptions& options);

	Ort::Env env;
	Ort::Session* session_;
};
