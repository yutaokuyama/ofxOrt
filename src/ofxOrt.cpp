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
