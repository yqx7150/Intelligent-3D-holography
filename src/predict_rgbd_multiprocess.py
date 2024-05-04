import os
from rtholo import *
from utils import *
from imageio.v2 import imread
from depthcamera_ctrl import Depthcamera_ctrl
import cv2
import onnx
import time
import torch
from getBlaze import getBlaze
import gxipy as gx
import multiprocessing
from trt import TRT





def load_IntensityImg(img):
    if len(img.shape) < 3:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    img = img[..., 0, np.newaxis]
    im = im2float(img, dtype=np.float32)  # convert to double, max 1

    im = torch.from_numpy(im).to("cuda")

    low_val = im <= 0.04045
    im[low_val] = 25 / 323 * im[low_val]  

    im[torch.logical_not(low_val)] = (
        (200 * im[torch.logical_not(low_val)] + 11) / 211  
    ) ** (12 / 5)
    
    amp = torch.sqrt(im)  # to amplitude
    amp = amp.cpu().numpy()
    amp = np.transpose(amp, axes=(2, 0, 1))
    amp = resize_keep_aspect(amp, [1024, 1024])
    # amp = pad_image(amp, [1024, 1024], padval=0, pytorch=False)
    amp = np.reshape(amp, (1, 1, 1024, 1024))
    amp = torch.from_numpy(amp)
    amp = amp.to("cuda")
    return amp

def load_DepthImg(depth_map):
    if len(depth_map.shape) < 3:
        depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
    depth_map = depth_map[..., 1, np.newaxis]
    depth_map = im2float(depth_map, dtype=np.float32)
    depth_map = np.transpose(depth_map, axes=(2, 0, 1))
    depth_map = resize_keep_aspect(depth_map, [1024, 1024])
    # depth_map = pad_image(depth_map, [1024, 1024], padval=0, pytorch=False)
    depth_map = np.reshape(depth_map, (1, 1, 1024, 1024))
    depth_map = torch.from_numpy(depth_map)
    depth_map = depth_map.to("cuda")
    return depth_map








def Recons_capture():
    cv2.namedWindow('Reconstruction', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Reconstruction", 1365, 1000)
    cv2.moveWindow('Reconstruction', 0, 0)
    cap = gx.CameraCap(0)
    while True:
        cap.clear_buffer()
        Reconstruction = cap.read()
        Reconstruction = Reconstruction[250:2816, 930:3438, :]
        cv2.imshow('Reconstruction', Reconstruction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        
def CGH():
    RGBD = Depthcamera_ctrl()
    depth_scale = RGBD.get_depth_scale()
    depth_max = 0.9 / depth_scale
    depth_min = 0.3 / depth_scale

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device('cuda')
    checkpoint1 = torch.load("/home/pat/code/djq/backup/src/checkpoints/CNN_1024_30/53.pth")     
    
    TRT_seleted = True
    fp16 = True
    print("TRT_seleted:"+str(TRT_seleted))


    if(fp16 == True):
        print("fp16")
        trtFile1 = "/home/pat/code/djq/rtholo/trt/file/network1_v1.plan"
        trtFile2 = "/home/pat/code/djq/rtholo/trt/file/network2_v1.plan"
    else:
        print("fp32")
        trtFile1 = "/home/pat/code/djq/rtholo/trt/file/network1_v1_fp32.plan"
        trtFile2 = "/home/pat/code/djq/rtholo/trt/file/network2_v1_fp32.plan"

    T = TRT(trtFile1, trtFile2)
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


    rtholo = rtholo(mode='test', feature_size = feature_size, size=img_size,img_distance =img_distance, distance_range=distance_range,
                        layers_num=layers_num,num_layers=num_layers, num_filters_per_layer=num_filters_per_layer).to(device)
    rtholo.load_state_dict(checkpoint1)
    rtholo.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    Blaze_phase = getBlaze(np.zeros([1024*2, 1024*2]), Blaze_phase_data, direction)
    Blaze_phase = torch.tensor(Blaze_phase, device="cuda") 
    
    cv2.namedWindow(' ', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(' ', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(' ', 1920, 0)
    cv2.resizeWindow(' ', 3840,2160)

    cv2.namedWindow('Align depth', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Align depth", 1280, 480)
    cv2.moveWindow('Align depth', 0, 0)

    holo_double = torch.zeros((2*img_size, 2*img_size), device=device)

    while True:
        starter.record()  
        depth, image = RGBD.get_frame()
        
        depth_3d = np.dstack((depth, depth, depth))
        between_img = np.where((depth_3d < depth_max) &
                            (depth_3d > depth_min), image, 0)
        between_depth = np.where((depth < depth_max) &
                                (depth > depth_min), depth, 0) #!
        depth_mask = np.where((depth < depth_max) & (depth > depth_min), 255, 0)
        between_depth = (between_depth - depth_min) / (depth_max - depth_min)*depth_mask
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(between_depth, alpha=1.0), cv2.COLORMAP_JET)
        img_show = np.hstack((between_img, depth_colormap))
        

        between_img = between_img[:,:,1]
        between_img = (between_img - between_img.min()) / (between_img.max() - between_img.min())
        between_depth = (between_depth - between_depth.min()) / (between_depth.max() - between_depth.min())
        
        amp = load_IntensityImg(between_img)
        depth_map = load_DepthImg(between_depth)


        input = torch.cat([amp, depth_map], dim=-3)

        with torch.no_grad():
            if TRT_seleted:
                # print("TRT")
                input = input.cpu().numpy()
                holo = T.inference(input)
            else:
                # print("no_TRT")
                holo, _, _ = rtholo(input, 0)   
        

        # hologram 
        holo = torch.squeeze(holo)
        holo = holo + torch.pi
        
        holo_double[0:2*img_size:2, 0:2*img_size:2] = holo
        holo_double[1:2*img_size:2, 0:2*img_size:2] = holo
        holo_double[0:2*img_size:2, 1:2*img_size:2] = holo
        holo_double[1:2*img_size:2, 1:2*img_size:2] = holo

        
        holo_out = torch.fmod(holo_double + Blaze_phase, 2*torch.pi)

        
        holo_out = pad_image(holo_out, [2160, 3840], padval=0, stacked_complex=False)
        
        holo_out = (holo_out / holo_out.amax(dim = (0,1)) * 255)
            
        holo_out1 = holo_out.type(torch.uint8).cpu().numpy()
        
        
        cv2.imshow('Align depth', img_show)
        cv2.imshow(' ',holo_out1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ender.record()
        torch.cuda.synchronize()       
        
        # print("Elapsed time: ", starter.elapsed_time(ender))
        #输出帧率
        print("FPS: {:.2f}".format(1 / (starter.elapsed_time(ender)/1000)))




if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    m1 = multiprocessing.Process(target=Recons_capture)
    m2 = multiprocessing.Process(target=CGH)
    m1.start()
    m2.start()


    
