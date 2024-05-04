import os
from imageio.v2 import imread
import cv2
import onnx
import time
import torch
from getBlaze import getBlaze
from trt import TRT

# 正常:52.22 ms TRT_fp16:14.48 ms TRT_fp32:21.05 ms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda')
checkpoint1 = torch.load("/home/pat/code/djq/backup/src/checkpoints/CNN_1024_30/53.pth")

TRT_seleted = True
print("TRT_seleted:"+str(TRT_seleted))

trtFile1 = "/home/pat/code/djq/rtholo/trt/file/network1_v1.plan"
trtFile2 = "/home/pat/code/djq/rtholo/trt/file/network2_v1.plan"
# trtFile1 = "/home/pat/code/djq/rtholo/trt/file/network1_v1_fp32.plan"
# trtFile2 = "/home/pat/code/djq/rtholo/trt/file/network2_v1_fp32.plan"

TRT = TRT(trtFile1, trtFile2)

CNNPP = False
channel = 0

layers_num=30

num_layers = 10
num_filters_per_layer=15 
feature_size= 7.48e-6


img_size = 1024
distance_range = 0.03
img_distance = 0.2

Blaze_phase_data = 0.5 * np.pi   #rightup 0.4
direction = 'up'

rtholo = rtholo(mode='test', feature_size = feature_size, size=img_size,img_distance =img_distance, distance_range=distance_range,
                     layers_num=layers_num,num_layers=num_layers, num_filters_per_layer=num_filters_per_layer, CNNPP=CNNPP).to(device)
rtholo.load_state_dict(checkpoint1)
rtholo.eval()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

img_idx = 2
depth_idx = 3
target_path = ("../dataset/example")

#       0             1     2           3       4             5                    6       7       8      9           10    11       12       13       14
img = ['bbb_rgb', '0886', 'bunny', 'castle', 'couch_rgb', 'USAF-1951-padding', 'point', 'grid', 'grid2', 'grid3', 'grid4', 'bbb', 'letter', 'color', 'long', 'frame_0043', 'amp1']
depth_char = ['bbb_depth', 'w', 'bunny_depth', 'b', 'couch_depth', 'USAF-1951-padding-depth', 'depth', 'letter_depth', 'colordepth', 'longdepth', 'frame_0043d', 'depth1']
#                0          1        2          3        4              5                        6          7               8           9
img_name = combine(img[img_idx], '.png')
depth_name = combine(depth_char[depth_idx], '.png')
all_name = combine(img[img_idx], depth_char[depth_idx])
img_path = os.path.join(target_path, img_name)
print(img_path)
depth_path = os.path.join(target_path, depth_name)

img = imread(img_path)

if len(img.shape) < 3:
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
img = img[..., channel, np.newaxis]
im = im2float(img, dtype=np.float32)  # convert to double, max 1
low_val = im <= 0.04045
im[low_val] = 25 / 323 * im[low_val]
im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                / 211) ** (12 / 5)
amp = np.sqrt(im)  # to amplitude
amp = np.transpose(amp, axes=(2, 0, 1))
amp = resize_keep_aspect(amp, [img_size, img_size])
target = np.reshape(amp, (img_size, img_size))
amp = np.reshape(amp, (1, 1, img_size, img_size))
amp = torch.from_numpy(amp)
amp = amp.to(device)
img_save = amp.cpu().squeeze().numpy()
img_save = (img_save - img_save.min()) / (img_save.max() - img_save.min())
img_save = ((img_save)*255).round().astype(np.uint8)

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
d_save = depth_map.cpu().squeeze().numpy()
d_save = (d_save - d_save.min()) / (d_save.max() - d_save.min())
d_save = ((d_save)*255).round().astype(np.uint8)

input = torch.cat([amp, depth_map], dim=-3)
source = pad_image(input, [img_size, img_size], padval=0, stacked_complex=False)
resoution = [img_size, img_size]

time_total = []

with torch.no_grad():
    for i in range(100):
        starter.record()
        if TRT_seleted:
            input = input.cpu().numpy()
            holo = TRT.inference(input)
        else:
            holo, _, _ = rtholo(input, 0) 
            input = input.cpu().numpy()
        ender.record()
        torch.cuda.synchronize()
        time = starter.elapsed_time(ender)
        if i != 0:
            time_total.append(time)
        print("{:.2f}".format(time), "ms") 
        input = torch.from_numpy(input).to(device)
print("average time: {:.2f}".format(np.mean(time_total)), "ms")

