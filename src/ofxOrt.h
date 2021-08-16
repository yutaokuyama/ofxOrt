#pragma once
#include "ofMain.h"
#include "ofxOrtTensors.h"
#include "ofxOrtUtils.h"
#include "onnxruntime_cxx_api.h"

class ofxOrt {
public:
  ofxOrt(const ORTCHAR_T *modelName, bool useCUDA);
  Ort::Session *getSession();

  void forward(const Ort::RunOptions &run_options,
               const char *const *input_names, const Ort::Value *input_values,
               size_t input_count, const char *const *output_names,
               Ort::Value *output_values, size_t output_count);

  const size_t getSessionInputCount();
  const size_t getSessionOutputCount();
  const string getSessionInputName();
  const string getSessionOutputName();

  const ONNXTensorElementDataType getSessionInputType();
  const ONNXTensorElementDataType getSessionOutputType();

  const std::vector<int64_t> getInputDims();
  const std::vector<int64_t> getOutputDims();

  void printModelInfo();

private:
  void enableCUDA(Ort::SessionOptions &options);

  Ort::Env env;
  Ort::Session *session_;
  Ort::AllocatorWithDefaultOptions defaultAllocator_;
};
