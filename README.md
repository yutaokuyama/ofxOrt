# ofxOrt
Thin wrapper of [ONNX runtime](https://onnxruntime.ai/) for openFrameworks.
[![Image from Gyazo](https://i.gyazo.com/56963392300c548b3499d25184385e7e.gif)](https://gyazo.com/56963392300c548b3499d25184385e7e)
## Note
**This addon is still in an experimental stage,** so I'll make breaking changes frequently :(.

## Prepare 
Install ONNX runtime by following the [document](https://onnxruntime.ai/docs/how-to/install.html).

## Usage
```C++
//First, create ofxOrt instance.
ofxOrt ofxOrt(ORT_TSTR("model.name");

//Create input and output tensors.
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
ofxOrtImageTensor<float> input_tensor(memory_info, content.getTexture());
ofxOrtImageTensor<float> output_tensor(memory_info, numChannels, outWidth, outHeight, true);

//Run inference.
ort.forward(Ort::RunOptions{ nullptr }, input_names, &(input_tensor.getTensor()), 1, output_names, &(output_tensor.getTensor()), 1);
```


## References
[ONNX Runtime Inference Examples](https://github.com/microsoft/onnxruntime-inference-examples)
[onnx_runtime_cpp](https://github.com/xmba15/onnx_runtime_cpp)
 
# License 
ofxOrt is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).