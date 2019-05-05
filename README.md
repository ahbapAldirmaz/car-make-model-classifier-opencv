# Car make and model classification using OpenCV - C++ example

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A C++ example for using [Spectrico's car make and model classifier](http://spectrico.com/car-make-model-recognition.html). Tested on Windows 10 and Ubuntu Linux 16.04 LTS. It takes as input a cropped single car image. The size doesn't matter. Because the car image is not always square, it is padded by the demo to become square and then it is resized to 224x224 pixels, so the aspect ratio is keeped. The demo returns the first 3 car models probabilities. The classifier is based on Mobilenet v2 (OpenCV backend). It takes 35 milliseconds on Intel Core i5-7600 CPU for single classification. The light version of the classifier is slightly less accuracy but is 4 times faster. It is optimized for speed and is recommended for edge devices. The demo doesn't include the classifier for car make and model recognition. It is a commercial product and is available for purchase at [http://spectrico.com/car-make-model-recognition.html](http://spectrico.com/car-make-model-recognition.html). A free version of the classifier with lower accuracy is available for download at [http://spectrico.com/pricing-car-mmr.html](http://spectrico.com/pricing-car-mmr.html).


![image](https://github.com/spectrico/car-make-model-classifier-opencv/blob/master/car-make-model.png?raw=true)

---

#### Usage
The demo is started using:
```
$ opencv_car_make_model_classifier car.jpg
```
The output is printed to the console:
```
  Inference time, ms: 53.9597
  Top 3 probabilities:
  make: Ferrari   model: 458      confidence: 95.3477 %
  make: Ferrari   model: 488      confidence: 1.48396 %
  make: Ferrari   model: F430     confidence: 0.526152 %
```

---
## Requirements
  - C++ compiler
  - OpenCV

---
## Credits
The car make and model classifier is based on MobileNetV2 mobile architecture: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
