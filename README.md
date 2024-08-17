## Solar PV Anomaly Detection



# Freezing Backbone Layers in Hybrid Model

To freeze the backbone layers of the Hybrid Model, which includes ResNet and Xception add the following lines of code after initializing ResNet and Xception:

```shell

for param in self.resnet.parameters():
    param.requires_grad = False


# Freeze Xception layers
for param in self.xception.parameters():
    param.requires_grad = False
```
# DeepStream Application

To run a DeepStream application on a Jetson device with optimized performance and speed, follow these steps:


To maximize the performance and power of your Jetson device, set the clocks to high:
```shell

sudo nvpmodel -m 0  # Set to MAX performance and power mode
sudo jetson_clocks  # Apply the high performance clock settings
```
Convert YOLOv10 to ONNX and TensorRT

1. Convert YOLOv10 (trained model) to ONNX format:
```shell

   python export.py --weights /path/to/yolov10_weights.pt --img-size 640 --batch-size 1 --device 0 --include onnx
```
2. Optimize the ONNX model with TensorRT and enable DLA (Deep Learning Accelerator) support:
```shell

   trtexec --onnx=/path/to/yolov10_weights.onnx --fp16 --dla-core=0 --saveEngine=/path/to/yolov10_weights.trt
```
3.Run DeepStream Application

To run the DeepStream application with the optimized TensorRT model, use the following command:
```shell

deepstream-app -c /path/to/config_file.txt

```

