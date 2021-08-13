#include "ofxOrt.h"

ofxOrt::ofxOrt(const ORTCHAR_T* modelName, bool useCUDA)
{
	try {
		Ort::SessionOptions sessionOption;
		if (useCUDA) {
			enableCUDA(sessionOption);
		}

		session_ = new Ort::Session(env, modelName, sessionOption);
	}
	catch (const Ort::Exception& exception) {
		std::cerr << exception.what() << std::endl;
	}

}

Ort::Session* ofxOrt::getSession() {
	return session_;
}

void ofxOrt::forward(const Ort::RunOptions& run_options, const char* const* input_names, const Ort::Value* input_values, size_t input_count,
	const char* const* output_names, Ort::Value* output_values, size_t output_count) {
	session_->Run(Ort::RunOptions{ nullptr }, input_names, input_values, input_count, output_names, output_values, output_count);
}

void ofxOrt::enableCUDA(Ort::SessionOptions& options) {
	OrtCUDAProviderOptions cuda_option = OrtCUDAProviderOptions{ 0 };
	options.AppendExecutionProvider_CUDA(cuda_option);
}


const size_t ofxOrt::getSessionInputCount() {
	assert(session_);
	return session_->GetInputCount();
}
const size_t ofxOrt::getSessionOutputCount() {
	assert(session_);
	return session_->GetOutputCount();
}
const string ofxOrt::getSessionInputName() {
	assert(session_);
	return session_->GetInputName(0, defaultAllocator_);
}
const string ofxOrt::getSessionOutputName() {
	assert(session_);
	return session_->GetOutputName(0, defaultAllocator_);
}

const ONNXTensorElementDataType ofxOrt::getSessionInputType() {
	assert(session_);
	Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
	return inputType;
}
const ONNXTensorElementDataType ofxOrt::getSessionOutputType() {
	assert(session_);
	Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
	ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
	return outputType;
}

const std::vector<int64_t> ofxOrt::getInputDims() {
	assert(session_);
	Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
	auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
	return inputTensorInfo.GetShape();

}
const std::vector<int64_t> ofxOrt::getOutputDims() {
	assert(session_);
	Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(0);
	auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
	return outputTensorInfo.GetShape();;
}

void ofxOrt::printModelInfo() {
	std::cout << "---------------------" << std::endl;
	std::cout << "Model Info:" << std::endl;
	std::cout << "Input Count: " << getSessionInputCount() << std::endl;
	std::cout << "Output Count: " << getSessionOutputCount() << std::endl;
	std::cout << "Input Name: " << getSessionInputName() << std::endl;
	std::cout << "Output Name: " << getSessionOutputName() << std::endl;
	std::cout << "Input Type: " << getSessionInputType() << std::endl;
	std::cout << "Output Type: " << getSessionOutputType() << std::endl;
	std::cout << "Input dims: ";
	for (auto& value : getInputDims()) {
		std::cout << value << " ";
	}
	std::cout << std::endl;
	std::cout << "Output dims: ";
	for (auto& value : getOutputDims()) {
		std::cout << value << " ";
	}
	std::cout << std::endl;
	std::cout << "---------------------" << std::endl;

}
