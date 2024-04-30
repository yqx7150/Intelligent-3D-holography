import cv2
import numpy as np
import tensorrt as trt
import torch as t
from cuda import cudart
import  torch
from utils import *
from propagation_ASM import *

feature_size = 7.48e-6
img_distance = 0.2

class TRT():
    def __init__(self, trtFile1, trtFile2) -> None:
        with open(trtFile1, "rb") as f:
            self.engine1 = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
                f.read()
            )
            if self.engine1 == None:
                print("Failed building engine1!")
                exit()
            print("Succeeded building engine1!")   
            
        self.nIO = self.engine1.num_io_tensors
        self.lTensorNames = [self.engine1.get_tensor_name(i) for i in range(self.nIO)]
        self.nInput = [self.engine1.get_tensor_mode(self.lTensorNames[i]) for i in range(self.nIO)].count(
            trt.TensorIOMode.INPUT
        )  
        self.context = self.engine1.create_execution_context()
        self.context.set_input_shape(self.lTensorNames[0], [1, 2, 1024, 1024])
        for i in range(self.nIO):
            print("Tensor %d: %s, %s" % (i, self.lTensorNames[i], str(self.engine1.get_binding_shape(i))))
            
        with open(trtFile2, "rb") as f:
            self.engine2 = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
                f.read()
            )
            if self.engine2 == None:
                print("Failed building engine2!")
                exit()
            print("Succeeded building engine2!")
        self.nIO2 = self.engine2.num_io_tensors
        self.lTensorNames2 = [self.engine2.get_tensor_name(i) for i in range(self.nIO2)]
        self.nInput2 = [self.engine2.get_tensor_mode(self.lTensorNames2[i]) for i in range(self.nIO2)].count(
            trt.TensorIOMode.INPUT
        )
        self.context2 = self.engine2.create_execution_context()
        self.context2.set_input_shape(self.lTensorNames2[0], [1, 2, 1024, 1024])
        for i in range(self.nIO2):
            print(
                "Tensor %d: %s, %s" % (i, self.lTensorNames2[i], str(self.engine2.get_binding_shape(i)))
            )
            
        self.precomputed_H = propagation_ASM(
            torch.empty(1, 1, 1024, 1024),
            feature_size = [feature_size, feature_size],
            wavelength=632e-9,
            z=img_distance,
            return_H=True,
        )
        self.precomputed_H = self.precomputed_H.to("cuda").detach()
        self.precomputed_H.requires_grad = False
        
    def inference(self, img):
            
        bufferH = []
        bufferH.append(np.ascontiguousarray(img))
        
        for i in range(self.nInput, self.nIO):
            bufferH.append(
                    np.empty(
                        self.context.get_tensor_shape(self.lTensorNames[i]),
                        dtype=trt.nptype(self.engine1.get_tensor_dtype(self.lTensorNames[i])),
                    )
                )
            
        bufferD = []
        for i in range(self.nIO):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(self.nInput):
                cudart.cudaMemcpy(
                    bufferD[i],
                    bufferH[i].ctypes.data,
                    bufferH[i].nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
        for i in range(self.nIO):
            self.context.set_tensor_address(self.lTensorNames[i], int(bufferD[i]))
        self.context.execute_async_v3(0)
        
        for i in range(self.nInput, self.nIO):
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
        slm_field = propagation_ASM(target_field, [feature_size,feature_size], 632e-9, img_distance, precomped_H=self.precomputed_H)
        slm_amp, slm_phase = rect_to_polar(slm_field.real, slm_field.imag)
        slm_field = torch.cat([slm_amp, slm_phase], dim=-3)
        
        bufferH2 = []
        bufferH2.append(np.ascontiguousarray(slm_field.cpu()))
        
        for i in range(self.nInput2, self.nIO2):
                bufferH2.append(
                    np.empty(
                        self.context2.get_tensor_shape(self.lTensorNames2[i]),
                        dtype=trt.nptype(self.engine2.get_tensor_dtype(self.lTensorNames2[i])),
                    )
                )
                
        bufferD2 = []
        for i in range(self.nIO2):
            bufferD2.append(cudart.cudaMalloc(bufferH2[i].nbytes)[1])
            
        for i in range(self.nInput2):
                cudart.cudaMemcpy(
                    bufferD2[i],
                    bufferH2[i].ctypes.data,
                    bufferH2[i].nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
                
        for i in range(self.nIO2):
            self.context2.set_tensor_address(self.lTensorNames2[i], int(bufferD2[i]))
            
        self.context2.execute_async_v3(0)
        
        for i in range(self.nInput2, self.nIO2):
                cudart.cudaMemcpy(
                    bufferH2[i].ctypes.data,
                    bufferD2[i],
                    bufferH2[i].nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
                
        for b in bufferD2:
                cudart.cudaFree(b)
        holo = torch.from_numpy(bufferH2[1])
        
        return holo
