## Freezing Backbone Layers in Hybrid Model

To freeze the backbone layers of the Hybrid Model, which includes ResNet and Xception, follow these steps after initializing the models. This will prevent the weights of these backbone layers from being updated during training.

Add the following lines of code after initializing ResNet and Xception:

```python
# Freeze ResNet layers
for param in self.resnet.parameters():
    param.requires_grad = False

# Freeze Xception layers
for param in self.xception.parameters():
    param.requires_grad = False

## DeepStream Application

To run a DeepStream application on a Jetson device with optimized performance and speed, follow these steps:


# Set Clocks to High Performance
sudo nvpmodel -m 0  # Set to MAX performance and power mode
sudo jetson_clocks  # Apply the high performance clock settings

# Convert the .pt (trained YOLOv10 model) file to ONNX and then to TensorRT engine with DLA Support
python -m torch.onnx.export <your_model> <model_output_path>.onnx --input_shapes <input_shapes>
trtexec --onnx=<model_output_path>.onnx --saveEngine=<model_output_path>.engine

# Run the DeepStream application with the specified configuration file
deepstream-app -c <path_to_your_config_file>.txt


