
[Device INFO](#Some-infomation-about-device)

[Usage](#How-to-use)

[Consumption](#The-consumption-of-resources-while-the-program-is-running(8-video-streams))

## Some infomation about device😄

- <font color="red">NVIDIA Jetson AGX Xavier 32GB</font>

![p9ZxbVJ.png](https://s1.ax1x.com/2023/04/23/p9ZxbVJ.png)

## How to use😮

```shell
mkdir build
cd build
cmake ..
make 
#run only one video stream
sudo ./yolo_det ../best.engine 1

#run 4 video streams
sudo ./yolo_det ../best.engine 4

#run 8 video streams
sudo ./yolo_det ../best.engine 8
NMS time: 0ms
Draw bounding boxes time: 0ms
ready to infer
readCvMat time: 85ms
readCvMat time: 93ms
readCvMat time: 98ms
readCvMat time: 100ms
readCvMat time: 100ms
readCvMat time: 101ms
readCvMat time: 101ms
readCvMat time: 102ms
Preprocess time: 102ms
inference time: 130ms
NMS time: 0ms
Draw bounding boxes time: 0ms
ready to infer
readCvMat time: 48ms
readCvMat time: 73ms
readCvMat time: 73ms
readCvMat time: 83ms
readCvMat time: 84ms
readCvMat time: 85ms
readCvMat time: 85ms
readCvMat time: 86ms
Preprocess time: 89ms
inference time: 134ms
NMS time: 0ms
Draw bounding boxes time: 0ms
ready to infer
readCvMat time: 74ms
readCvMat time: 78ms
readCvMat time: 82ms
readCvMat time: 82ms
readCvMat time: 83ms
readCvMat time: 84ms
readCvMat time: 85ms
readCvMat time: 85ms
Preprocess time: 87ms
inference time: 132ms
NMS time: 0ms
Draw bounding boxes time: 0ms
ready to infer
readCvMat time: 70ms
readCvMat time: 85ms
readCvMat time: 85ms
readCvMat time: 85ms
readCvMat time: 86ms
readCvMat time: 88ms
readCvMat time: 89ms
readCvMat time: 89ms
Preprocess time: 91ms
inference time: 131ms
NMS time: 0ms
Draw bounding boxes time: 0ms
ready to infer
readCvMat time: 86ms
readCvMat time: 88ms
readCvMat time: 88ms
readCvMat time: 91ms
readCvMat time: 96ms
readCvMat time: 96ms
readCvMat time: 97ms
readCvMat time: 98ms
Preprocess time: 100ms
inference time: 137ms
NMS time: 0ms
Draw bounding boxes time: 0ms
ready to infer
readCvMat time: 87ms
readCvMat time: 94ms
readCvMat time: 96ms
readCvMat time: 100ms
readCvMat time: 101ms
readCvMat time: 101ms
readCvMat time: 102ms
readCvMat time: 102ms
Preprocess time: 109ms
inference time: 146ms
NMS time: 0ms
Draw bounding boxes time: 0ms
ready to infer
readCvMat time: 41ms
readCvMat time: 77ms
readCvMat time: 79ms
readCvMat time: 89ms
readCvMat time: 90ms
readCvMat time: 92ms
readCvMat time: 93ms
readCvMat time: 93ms
Preprocess time: 96ms
inference time: 133ms
```



## The consumption of resources while the program is running(8 video streams)🚀

