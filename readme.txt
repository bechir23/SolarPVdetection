To Freeze the backbone layers, add these lines in the hybrid model after initializing ResNet and Xception:
for param in self.resnet.parameters():
    param.requires_grad = False

for param in self.xception.parameters():
    param.requires_grad = False
