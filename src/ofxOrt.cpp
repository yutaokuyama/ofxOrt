#include "ofxOrt.h"

ofxOrt::ofxOrt(const ORTCHAR_T *modelName, bool useCUDA) {
  try {
    Ort::SessionOptions sessionOption;
    if (useCUDA) {
      enableCUDA(sessionOption);
    }

    session_ = new Ort::Session(env, modelName, sessionOption);
  } catch (const Ort::Exception &exception) {
    std::cerr << exception.what() << std::endl;
  }
}

Ort::Session *ofxOrt::getSession() { return session_; }

void ofxOrt::forward(const Ort::RunOptions &run_options,
                     const char *const *input_names,
                     const Ort::Value *input_values, size_t input_count,
                     const char *const *output_names, Ort::Value *output_values,
                     size_t output_count) {
  session_->Run(Ort::RunOptions{nullptr}, input_names, input_values,
                input_count, output_names, output_values, output_count);
}

void ofxOrt::enableCUDA(Ort::SessionOptions &options) {
  OrtCUDAProviderOptions cuda_option = OrtCUDAProviderOptions{0};
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
std::string ofxOrt::getSessionInputName() const {
  assert(session_);
  Ort::AllocatorWithDefaultOptions defaultAllocator;
  return session_->GetInputName(0, defaultAllocator);
}
std::string ofxOrt::getSessionOutputName() const {
  assert(session_);
  Ort::AllocatorWithDefaultOptions defaultAllocator;
  return session_->GetOutputName(0, defaultAllocator);
}

ONNXTensorElementDataType ofxOrt::getSessionInputType() const {
  assert(session_);
  Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
  return inputType;
}
ONNXTensorElementDataType ofxOrt::getSessionOutputType() const {
  assert(session_);
  Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
  return outputType;
}

std::vector<int64_t> ofxOrt::getInputDims() const {
  assert(session_);
  Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  return inputTensorInfo.GetShape();
}
std::vector<int64_t> ofxOrt::getOutputDims() const {
  assert(session_);
  Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
  return outputTensorInfo.GetShape();
}

void ofxOrt::printModelInfo() const {
  std::cout << "---------------------" << std::endl;
  std::cout << "Model Info:" << std::endl;
  std::cout << "Input Count: " << getSessionInputCount() << std::endl;
  std::cout << "Output Count: " << getSessionOutputCount() << std::endl;
  std::cout << "Input Name: " << getSessionInputName() << std::endl;
  std::cout << "Output Name: " << getSessionOutputName() << std::endl;
  std::cout << "Input Type: "
            << ofxOrtUtils::getONNXTensorElementDataTypeName(
                   getSessionInputType())
            << std::endl;
  std::cout << "Output Type: "
            << ofxOrtUtils::getONNXTensorElementDataTypeName(
                   getSessionOutputType())
            << std::endl;
  std::cout << "Input dims: ";
  for (auto value : getInputDims()) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
  std::cout << "Output dims: ";
  for (auto value : getOutputDims()) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
  std::cout << "---------------------" << std::endl;
}
