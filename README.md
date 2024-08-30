## Solar PV Anomaly Detection

# Flash Jetson Orin NX with JetPack 5.1.2


Step 1:Install the required dependencies on your Ubuntu host PC:
```shell

 sudo apt install qemu-user-static sshpass abootimg nfs-kernel-server libxml2-utils binutils -y
```

Step 2: Disable the USB autosuspend
```shell
 sudo sh -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend'
```

Step 3: Download and Extract NVIDIA Drivers

Download the necessary NVIDIA drivers for L4T 35.4.1 on the host PC.

Extract the Jetson Linux package and the Root Filesystem package:
```shell
 tar -xjf Jetson_Linux_R35.4.1_aarch64.tbz2
 sudo tar -xjf Tegra_Linux_Sample-Root-Filesystem_R35.4.1_aarch64.tbz2 -C Linux_for_Tegra/rootfs/
```

Navigate to the Linux_for_Tegra directory:
```shell

 cd Linux_for_Tegra/
```

Apply binaries and install flash prerequisites:
```shell

 sudo ./apply_binaries.sh
 sudo ./tools/l4t_flash_prerequisites.sh
```

Navigate to the bootloader directory and apply NVIDIA patch for JP5.1.2:
```shell

 cd bootloader/t186ref/BCT
```

Edit the tegra234-mb2-bct-scr-p3767-0000.dts file and add the following lines under the tfc section:
```shell

 tfc {
    reg@322 { /* GPIO_M_SCR_00_0 */
    exclusion-info = <2>;
    value = <0x38008080>;
    };
 };
```

Step 4: Pre-configure Username, Password, and Hostname
```shell


 cd Linux_for_Tegra
 sudo tools/l4t_create_default_user.sh -u bechir -p bechir -a -n bechir-desktop --accept-license
```

Step 5: Flash the Jetson Device to the NVMe SSD
```shell


 sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device nvme0n1p1 \
  -c tools/kernel_flash/flash_l4t_external.xml -p "-c bootloader/t186ref/cfg/flash_t234_qspi.xml" \
  --showlogs --network usb0 p3509-a02+p3767-0000 internal

```


# ReComputer J4012

![Photo](JetsonModule.png)


# Freezing Backbone Layers in Hybrid Model

To freeze the backbone layers of the Hybrid Model, which includes ResNet and Xception add the following lines of code after initializing ResNet and Xception:
```shell
# Freeze Resnet layers
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

   python3 export_yoloV10.py --weights IRdetection.pt
```
## Inference on DLA (Deep Learning Accelerator) and GPU
To configure deepstream to run inference on both DLAs and GPU in a parallel mode in a single process :

2. Optimize the ONNX model with TensorRT and enable DLA support:
```shell

   trtexec --onnx=IRdetection.onnx --fp16 --useDLACore=0 --saveEngine=IRdetection.engine --allowGPUFallback
```
Or add the following lines in config_infer_primary_yoloV10.txt to achieve the equivalent configuration:
```shell
 [proprety]
 enable-dla=1
 use-dla-core=0
 
```
To analyze and dump the detailed information about the layers in TensorRT engine use :
```shell

  trtexec --loadEngine=IRdetection.engine --dumpLayerInfo

```

3. Run DeepStream Application

To run the DeepStream application with the optimized TensorRT model, use the following command:
```shell

deepstream-app -c deepstream_app_config.txt

```


## Inference on GPU
 1.Disable the model-engine line in config_infer_primary_yoloV10.txt to allow DeepStream to automatically create an engine using only the GPU (you can choose the network mode as you prefer):
 ```shell
 #model-engine=IRdetection.engine
```
 2.Run deepstream-app with specified configuration :
```shell

deepstream-app -c deepstream_app_config.txt

```

## INT8 Calibration
1.Create a calibration Folder and Transfer Images

```shell
 mkdir calibration
 for jpg in $(ls -1 PVsolar/train/images/*.jpg | sort -R | head -1000); do \
     cp ${jpg} calibration/; \
 done
```
2.Create file calibration.txt with images paths
```shell
  realpath calibration/*jpg > calibration.txt
```
3.Set Environment Variables 
```shell
 export INT8_CALIB_IMG_PATH=calibration.txt
 export INT8_CALIB_BATCH_SIZE=8
```
4.Quantize with tensorRT 
```shell
 cd DeepStream-Yolo

 trtexec --onnx=IRdetection.onnx --int8 --calib=calibration.txt --saveEngine=IRdetection_b8_gpu_int8.engine

```


# TritonServer
For deploying Triton Inference Server on NVIDIA Jetson device (iGPU) follow these steps to avoid a non detect GPU devices error :

1. Pull the Triton Inference Server Docker image:
```shell
   docker pull nvcr.io/nvidia/tritonserver:23.11-py3-igpu

```
3. Run the Triton Inference Server Docker container:
```shell
 docker run --runtime=nvidia -v /home/bechir/models:/models -it nvcr.io/nvidia/tritonserver:23.11-py3-igpu bash


```
4. Navigate to the Triton server directory and start the server:
```shell
 cd /opt/tritonserver
 ./tritonserver --model-repository=/models

```

