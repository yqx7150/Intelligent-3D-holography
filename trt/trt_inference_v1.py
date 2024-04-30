import os
from datetime import datetime as dt
from glob import glob

import cv2
import numpy as np
import tensorrt as trt
import torch as t
import torch.nn.functional as F
from cuda import cudart
from torch.autograd import Variable
import sys
from imageio.v2 import imread


sys.path.append("/home/pat/code/djq/rtholo/src")
from rtholo import *
from getBlaze import *

from utils import *
from propagation_ASM import *
import time

import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)

# from heartrate import trace
# trace(browser=True)


channel = 1

layers_num=100

num_layers = 10
num_filters_per_layer=15 
feature_size= 7.48e-6


img_size = 1024
distance_range = 0.03
img_distance = 0.2

Blaze_phase_data = 0.5*np.pi
direction = 'up'
device = 'cuda'

video_path = "/home/pat/code/djq/rtholo/src/bbb_sunflower_1080p_30fps_normal.mp4"
depth_path = "/home/pat/code/djq/rtholo/dataset/example/b.png"

trtFile1 = "/home/pat/code/djq/rtholo/trt/file/network1_v1.plan"
trtFile2 = "/home/pat/code/djq/rtholo/trt/file/network2_v1.plan"


amp_ones = torch.ones([img_size,img_size]).cuda()

starter, ender = t.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def load_IntensityImg(img):
    if len(img.shape) < 3:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    img = img[..., channel, np.newaxis]
    im = im2float(img, dtype=np.float32)  # convert to double, max 1

    im = torch.from_numpy(im).to(device)

    low_val = im <= 0.04045
    im[low_val] = 25 / 323 * im[low_val]  

    im[torch.logical_not(low_val)] = (
        (200 * im[torch.logical_not(low_val)] + 11) / 211  
    ) ** (12 / 5)
    
    amp = torch.sqrt(im)  # to amplitude
    amp = amp.cpu().numpy()
    amp = np.transpose(amp, axes=(2, 0, 1))
    amp = resize_keep_aspect(amp, [img_size, img_size])
    amp = np.reshape(amp, (1, 1, img_size, img_size))
    amp = torch.from_numpy(amp)
    amp = amp.to(device)
    return amp

def load_DepthImg(depth_path):
    depth_map = imread(depth_path)
    if len(depth_map.shape) < 3:
        depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
    depth_map = depth_map[..., 1, np.newaxis]
    depth_map = im2float(depth_map, dtype=np.float32)
    depth_map = np.transpose(depth_map, axes=(2, 0, 1))
    depth_map = resize_keep_aspect(depth_map, [img_size, img_size])
    depth_map = np.reshape(depth_map, (1, 1, img_size, img_size))
    depth_map = torch.from_numpy(depth_map)
    depth_map = depth_map.to(device)
    return depth_map


with open(trtFile1, "rb") as f:
    engine1 = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        f.read()
    )
    if engine1 == None:
        print("Failed building engine1!")
        exit()
    print("Succeeded building engine1!")

nIO = engine1.num_io_tensors
lTensorNames = [engine1.get_tensor_name(i) for i in range(nIO)]
nInput = [engine1.get_tensor_mode(lTensorNames[i]) for i in range(nIO)].count(
    trt.TensorIOMode.INPUT
)

context = engine1.create_execution_context()
context.set_input_shape(lTensorNames[0], [1, 2, img_size, img_size])
for i in range(nIO):
    print("Tensor %d: %s, %s" % (i, lTensorNames[i], str(engine1.get_binding_shape(i))))

with open(trtFile2, "rb") as f:
    engine2 = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        f.read()
    )
    if engine2 == None:
        print("Failed building engine2!")
        exit()
    print("Succeeded building engine2!")

nIO2 = engine2.num_io_tensors
lTensorNames2 = [engine2.get_tensor_name(i) for i in range(nIO2)]
nInput2 = [engine2.get_tensor_mode(lTensorNames2[i]) for i in range(nIO2)].count(
    trt.TensorIOMode.INPUT
)

context2 = engine2.create_execution_context()
context2.set_input_shape(lTensorNames2[0], [1, 2, img_size, img_size])
for i in range(nIO2):
    print(
        "Tensor %d: %s, %s" % (i, lTensorNames2[i], str(engine2.get_binding_shape(i)))
    )


precomputed_H = propagation_ASM(
    torch.empty(1, 1, 1024, 1024),
    feature_size=[feature_size,feature_size],
    wavelength=632e-9,
    z=img_distance,
    return_H=True,
)
precomputed_H = precomputed_H.to("cuda").detach()
precomputed_H.requires_grad = False

Blaze_phase = getBlaze(np.zeros([img_size*2, img_size*2]), Blaze_phase_data, direction)
Blaze_phase = torch.tensor(Blaze_phase, device=device) 

depth_map = load_DepthImg(depth_path)
cv2.namedWindow(' ', cv2.WINDOW_NORMAL)
cv2.setWindowProperty(' ', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.moveWindow(' ', 1920, 0)
cv2.resizeWindow(' ', 3840,2160)

holo_double = torch.zeros((2*img_size, 2*img_size), device=device)

while True:
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        for i in range(3):
            ret, frame = cap.read()
        if ret:
            starter.record()

            amp = load_IntensityImg(frame)
            input = torch.cat([amp, depth_map], dim=-3)
            input = input.cpu().numpy()

            with torch.no_grad():
                bufferH = []
                
                bufferH.append(np.ascontiguousarray(input))

                for i in range(nInput, nIO):
                    bufferH.append(
                        np.empty(
                            context.get_tensor_shape(lTensorNames[i]),
                            dtype=trt.nptype(engine1.get_tensor_dtype(lTensorNames[i])),
                        )
                    )

                bufferD = []
                for i in range(nIO):
                    bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

                for i in range(nInput):
                    cudart.cudaMemcpy(
                        bufferD[i],
                        bufferH[i].ctypes.data,
                        bufferH[i].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )

                for i in range(nIO):
                    context.set_tensor_address(lTensorNames[i], int(bufferD[i]))

                context.execute_async_v3(0)

                for i in range(nInput, nIO):
                    cudart.cudaMemcpy(
                        bufferH[i].ctypes.data,
                        bufferD[i],
                        bufferH[i].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )

                for b in bufferD:
                    cudart.cudaFree(b)

                target_amp, target_phase = bufferH[2], bufferH[1]
                target_amp, target_phase = (
                    torch.from_numpy(target_amp).cuda(),
                    torch.from_numpy(target_phase).cuda(),
                )

                obj_r, obj_i = polar_to_rect(target_amp, target_phase)
                target_field = torch.complex(obj_r, obj_i)

                slm_field = propagation_ASM(
                    target_field, [feature_size,feature_size], 632e-9, 0.2, precomped_H=precomputed_H
                )

                slm_amp, slm_phase = rect_to_polar(slm_field.real, slm_field.imag)
                slm_field = torch.cat([slm_amp, slm_phase], dim=-3)

                ###############!net2########################

                bufferH2 = []
                bufferH2.append(np.ascontiguousarray(slm_field.cpu()))

                for i in range(nInput2, nIO2):
                    bufferH2.append(
                        np.empty(
                            context2.get_tensor_shape(lTensorNames2[i]),
                            dtype=trt.nptype(engine2.get_tensor_dtype(lTensorNames2[i])),
                        )
                    )

                bufferD2 = []
                for i in range(nIO2):
                    bufferD2.append(cudart.cudaMalloc(bufferH2[i].nbytes)[1])

                for i in range(nInput2):
                    cudart.cudaMemcpy(
                        bufferD2[i],
                        bufferH2[i].ctypes.data,
                        bufferH2[i].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )

                for i in range(nIO2):
                    context2.set_tensor_address(lTensorNames2[i], int(bufferD2[i]))

                context2.execute_async_v3(0)

                for i in range(nInput2, nIO2):
                    cudart.cudaMemcpy(
                        bufferH2[i].ctypes.data,
                        bufferD2[i],
                        bufferH2[i].nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )


                for b in bufferD2:
                    cudart.cudaFree(b)
                
            # # hologram 
            holo = torch.squeeze(torch.from_numpy(bufferH2[1]))
            holo = holo + torch.pi
            
            holo_double[0:2*img_size:2, 0:2*img_size:2] = holo
            holo_double[1:2*img_size:2, 0:2*img_size:2] = holo
            holo_double[0:2*img_size:2, 1:2*img_size:2] = holo
            holo_double[1:2*img_size:2, 1:2*img_size:2] = holo


            holo_out = torch.fmod(holo_double + Blaze_phase, 2*torch.pi)


            holo_out = pad_image(holo_out, [2160, 3840], padval=0, stacked_complex=False)
            holo_out = (holo_out / holo_out.amax(dim = (0,1)) * 255)

            holo_out = np.uint8(holo_out.cpu().numpy())

            cv2.imshow(' ',holo_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ender.record()
            torch.cuda.synchronize()
            print("Elapsed time: ", starter.elapsed_time(ender))

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    break

