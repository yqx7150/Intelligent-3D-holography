# **Real-time intelligent 3D holographic photography for real-world scenarios**

Xianlin Song, Jiaqing Dong, Minghao Liu, Zehao Sun, Zibang Zhang, Jianghao Xiong, Zilong Li, Xuan Liu, Qiegen Liu*
Real-time intelligent 3D holographic photography for real-world scenarios       
Optics Express Vol. 32, Issue 14, pp. 24540-24552 (2024)      
https://doi.org/10.1364/OE.529107    

https://github.com/djq-2000/123/assets/56143723/b8c3cbd7-5bac-45f7-ad20-8a40451dd00d

## Getting Started

This code runs with Python 3.8.17, Pytorch 2.0.1 and TensorRT 8.6.0

**- [./src/](./src/)**
- [train.py](./src/train.py): The training code of the model
- [NET1.py](./src/NET1.py): The network structure of the model1
- [dataLoader.py](./src/dataLoader.py): The data loader of the model
- [rtholo.py](./src/rtholo.py): The code of the real-time holography
- [predict_rgbd_multiprocess.py](./src/predict_rgbd_multiprocess.py): The testing code of the model
- [trt.py](./src/trt.py): The code of TensorRT class
- [getBlaze.py](./src/getBlaze.py): This code is for generating a blazed grating
- [GCD_ctrl.py](./src/GCD_ctrl.py): This code is for controlling the motorized linear stage
- [depthcamera_ctrl.py](./src/depthcamera_ctrl.py): This code is for controlling the depth camera Realsense D435
- [gxipy](./src/gxipy): The SDK of the Daheng camera

**- [./trt/](./trt/)**
- [trt_create_v1.py](./trt/trt_create_v1.py): This code is used to generate the TRT model
- [trt_inference_v1.py](./trt/trt_inference_v1.py): This code is used to test the TRT model

## Training
```
python ./src/train.py --p_loss --l2_loss --num_epochs 60 --data_path <The address of your training set>
```

## Testing
```
python predict_rgbd_multiprocess.py
```

## Checkpoints
We provide pretrained checkpoints. The pre-trained models in  - [**./src/checkpoints/CNN_1024_30/53.pth**](./src/checkpoints/CNN_1024_30/53.pth)

## Ackonwledgement

We are thankful for the open source of **[tensor_holography
](https://github.com/liangs111/tensor_holography/tree/main)**,**[HoloEncoder](https://github.com/THUHoloLab/Holo-encoder)**, **[HoloEncoder-Pytorch-Version](https://github.com/flyingwolfz/holoencoder-python-version)** and **[Self-Holo](https://github.com/SXHyeah/Self-Holo)**.
These works are very helpful for our research.
