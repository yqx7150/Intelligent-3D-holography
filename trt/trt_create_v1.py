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

sys.path.append("/home/pat/code/djq/rtholo/src")
from rtholo import *

img_size = 1024
layers_num = 30
distance_range = 0.03
img_index = 1
channel = 1
num_filters_per_layer = 15
num_layers = 10

bUseFP16Mode = False
onnxFile1 = "./file/network1_v1.onnx"
onnxFile2 = "./file/network2_v1.onnx"
trtFile1 = "./file/network1_v1_fp32.plan"
trtFile2 = "./file/network2_v1_fp32.plan"
CNNPP = False


def onnx2trt(onnx_path, trt_path, bUseFP16Mode=False):
    # Parse network, rebuild network and do inference in TensorRT ------------------
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if bUseFP16Mode:
        config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_path):
        print("Failed finding ONNX file!")
        exit()
    print("Succeeded finding ONNX file!")
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    inputTensor = network.get_input(0)
    profile.set_shape(
        inputTensor.name,
        [1, 2, img_size, img_size],
        [1, 2, img_size, img_size],
        [1, 2, img_size, img_size],
    )
    config.add_optimization_profile(profile)

    # network.unmark_output(network.get_output(0))  # remove output tensor "y"
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")

    with open(trt_path, "wb") as f:
        f.write(engineString)


device = torch.device("cuda")
# checkpoint1 = torch.load("/home/pat/code/djq/rtholo/src/checkpoints/CNN_1024_30/82.pth")    
checkpoint1 = torch.load("/home/pat/code/djq/backup/src/checkpoints/CNN_1024_30/53.pth") 
self_holo = rtholo(
    size=img_size,
    distance_range=distance_range,
    layers_num=layers_num,
    num_filters_per_layer=num_filters_per_layer,
    num_layers=num_layers,
    CNNPP = CNNPP
).to(device)
self_holo.load_state_dict(checkpoint1)

self_holo.eval()

t.onnx.export(
    self_holo.get_Network1(),
    t.randn(1, 2, img_size, img_size, device="cuda"),
    onnxFile1,
    input_names=["rgb_d"],
    output_names=["amp", "phase"],
    do_constant_folding=True,
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=12,
    dynamic_axes={
        "rgb_d": {0: "nBatchSize"},
        "amp": {0: "nBatchSize"},
        "phase": {0: "nBatchSize"},
    },
)
print("Succeeded converting network1 into ONNX!")

t.onnx.export(
    self_holo.get_Network2(),
    t.randn(1, 2, img_size, img_size, device="cuda"),
    onnxFile2,
    input_names=["complex_amp"],
    output_names=["holo"],
    do_constant_folding=True,
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=12,
    dynamic_axes={"complex_amp": {0: "nBatchSize"}, "holo": {0: "nBatchSize"}},
)
print("Succeeded converting network2 into ONNX!")

onnx2trt(onnxFile1, trtFile1, bUseFP16Mode)
print("Succeeded converting network1 into TensorRT!")
onnx2trt(onnxFile2, trtFile2, bUseFP16Mode)
print("Succeeded converting network2 into TensorRT!")
