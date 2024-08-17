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
