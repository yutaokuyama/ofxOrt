# Object detection
[![Image from Gyazo](https://i.gyazo.com/c8392f5a4968c8dde5dd353079103c76.png)](https://gyazo.com/c8392f5a4968c8dde5dd353079103c76)

Demo of object detection by tiny-yolov2.
Please download ONNX model from [here](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov2).

I used this [repository](https://github.com/xmba15/onnx_runtime_cpp/tree/master/examples) as a reference for the parser of the results.
## Note
I have not removed the bounding box based on the confidence level, so an excess box will be displayed.