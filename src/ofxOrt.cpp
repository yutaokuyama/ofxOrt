#include "ofxOrt.h"

ofxOrt::ofxOrt(const ORTCHAR_T* modelName, bool useCUDA) {
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

Ort::Session* ofxOrt::getSession() { return session_; }

void ofxOrt::forward(const Ort::RunOptions& run_options,
	const char* const* input_names,
	const Ort::Value* input_values, size_t input_count,
	const char* const* output_names, Ort::Value* output_values,
	size_t output_count) {
	session_->Run(Ort::RunOptions{ nullptr }, input_names, input_values,
		input_count, output_names, output_values, output_count);
}

std::vector<Ort::Value> ofxOrt::forward(const Ort::RunOptions& run_options,
	const char* const* input_names, const Ort::Value* input_values,
	size_t input_count, const char* const* output_names,
	size_t output_count) {
	auto result = session_->Run(Ort::RunOptions{ nullptr }, input_names, input_values,
		input_count, output_names, output_count);
	return result;
}

void ofxOrt::enableCUDA(Ort::SessionOptions& options) {
	OrtCUDAProviderOptions cuda_option = OrtCUDAProviderOptions();
	options.AppendExecutionProvider_CUDA(cuda_option);
}

size_t ofxOrt::getSessionInputCount() const {
	assert(session_);
	return session_->GetInputCount();
}
size_t ofxOrt::getSessionOutputCount() const {
	assert(session_);
	return session_->GetOutputCount();
}
std::vector<std::string> ofxOrt::getSessionInputNames(size_t inputCount) const {
	assert(session_);
	Ort::AllocatorWithDefaultOptions defaultAllocator;
	std::vector<std::string> inputNames;
	std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
	for (int i = 0; i < inputCount; i++) {
		auto input_name = session_->GetInputNameAllocated(inputCount - i - 1, defaultAllocator);
		inputNodeNameAllocatedStrings.push_back(std::move(input_name));
		inputNames.push_back(inputNodeNameAllocatedStrings.back().get());
	}
	return inputNames;
}
std::vector<std::string> ofxOrt::getSessionOutputNames(size_t outputCount) const {
	assert(session_);
	Ort::AllocatorWithDefaultOptions defaultAllocator;
	std::vector<std::string> outputNames;
	std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
	for (int i = 0; i < outputCount; i++) {
		auto output_name = session_->GetOutputNameAllocated(outputCount - i - 1, defaultAllocator);
		outputNodeNameAllocatedStrings.push_back(std::move(output_name));
		outputNames.push_back(outputNodeNameAllocatedStrings.back().get());
	}
	return outputNames;
}

std::vector<ONNXTensorElementDataType> ofxOrt::getSessionInputTypes(size_t inputCount) const {
	assert(session_);
	std::vector<ONNXTensorElementDataType> inputTypes;
	for (int i = 0; i < inputCount; i++) {
		int index = inputCount - i - 1;
		Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(index);
		auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
		inputTypes.push_back(inputType);
	}

	return inputTypes;
}
std::vector<ONNXTensorElementDataType> ofxOrt::getSessionOutputTypes(size_t outputCount) const {
	assert(session_);
	std::vector<ONNXTensorElementDataType> outputTypes;
	for (int i = 0; i < outputCount; i++) {
		int index = outputCount - i - 1;
		Ort::TypeInfo inputTypeInfo = session_->GetOutputTypeInfo(index);
		auto outputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
		outputTypes.push_back(outputType);
	}
	return outputTypes;
}

std::vector<std::vector<int64_t>> ofxOrt::getInputDims(size_t inputCount) const {
	assert(session_);
	std::vector<std::vector<int64_t>> inputDims;
	for (int i = 0; i < inputCount; i++) {
		int index = inputCount - i - 1;
		Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(index);
		auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		inputDims.push_back(inputTensorInfo.GetShape());
	}
	return inputDims;
}


std::vector<std::vector<int64_t>> ofxOrt::getOutputDims(size_t outputCount) const {
	assert(session_);

	std::vector<std::vector<int64_t>> outputDims;
	for (int i = 0; i < outputCount; i++) {
		int index = outputCount - i - 1;
		Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(index);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		outputDims.push_back(outputTensorInfo.GetShape());
	}
	return outputDims;
}



void ofxOrt::printModelInfo() const {
	std::cout << "---------------------" << std::endl;
	std::cout << "Model Info:" << std::endl;
	std::cout << "Input Count: " << getSessionInputCount() << std::endl;
	std::cout << "Output Count: " << getSessionOutputCount() << std::endl;
	std::cout << "Input Names: { ";
	for (auto name : getSessionInputNames(getSessionInputCount())) {

		std::cout << name << ", ";
	};
	std::cout << " }" << std::endl;

	std::cout << "Output Names: { ";
	for (auto name : getSessionOutputNames(getSessionOutputCount())) {
		std::cout << name << ", ";
	}
	std::cout << " }" << std::endl;


	std::cout << "Input Types: { ";
	for (auto name : getSessionInputTypes(getSessionInputCount())) {
		std::cout << ofxOrtUtils::getONNXTensorElementDataTypeName((name)) << ", ";
	}
	std::cout << "}" << std::endl;


	std::cout << "Input Types: { ";
	for (auto name : getSessionOutputTypes(getSessionOutputCount())) {
		std::cout << ofxOrtUtils::getONNXTensorElementDataTypeName((name)) << ", ";
	}
	std::cout << "}" << std::endl;


	std::cout << "Input dims: ";
	for (auto dims : getInputDims(getSessionInputCount())) {
		std::cout << "{ ";
		for (auto dim : dims) {
			std::cout << dim << " ";
		}
		std::cout << "}, ";
	}
	std::cout << std::endl;

	std::cout << "Output dims:{ ";
	for (auto dims : getOutputDims(getSessionOutputCount())) {
		std::cout << "{ ";
		for (auto dim : dims) {
			std::cout << dim << ", ";
		}
		std::cout  << "}, ";
	}
	std::cout << "}" << std::endl;
	std::cout << std::endl;
	std::cout << "---------------------" << std::endl;
}