
[Device INFO](#Some-infomation-about-device)

[Usage](#How-to-use)

[Consumption](#The-consumption-of-resources-while-the-program-is-running(8-video-streams))

## Some infomation about device😄

- <font color="red">NVIDIA Jetson Orin Nano 8GB</font>

![device.png](./imgs/device.png)

## How to use😮

```shell
mkdir build
cd build
cmake ..
make 

#run 8 video streams
sudo ./yolov5_det ../weights/best.engine 

cap.read time: 15ms
Preprocess time: 14ms
inference time: 98ms
NMS time: 0ms
Draw bounding boxes time: 0ms
cap.read time: 3ms
cap.read time: 4ms
cap.read time: 8ms
cap.read time: 12ms
cap.read time: 14ms
cap.read time: 18ms
cap.read time: 3ms
cap.read time: 11ms
Preprocess time: 12ms
inference time: 93ms
NMS time: 0ms
Draw bounding boxes time: 0ms
cap.read time: 8ms
cap.read time: 12ms
cap.read time: 17ms
cap.read time: 14ms
cap.read time: 18ms
cap.read time: 19ms
cap.read time: 8ms
cap.read time: 19ms
Preprocess time: 11ms
inference time: 92ms
NMS time: 0ms
Draw bounding boxes time: 0ms
```



## The consumption of resources while the program is running(8 video streams)🚀

![top](./imgs/top.png)
![GPU](./imgs/GPU.png)
