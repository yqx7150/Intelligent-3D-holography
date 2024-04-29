import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import logging
import cv2 as cv
import torch.nn.functional as F
import time


def imwrite(img, savePath):
    img = img * 255
    cv2.imwrite(savePath, img)

def combine(*args):
    combined_str = []
    for arg in args:
        combined_str.append(str(arg))
    combined_str = "".join(combined_str)
    return combined_str

def ifftshift(tensor):
    """ifftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[3] / 2.0), 3)
    return tensor_shifted

def fftshift(tensor):
    """fftshift for tensors of dimensions [minibatch_size, num_channels, height, width, 2]

    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[2] / 2.0), 2)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[3] / 2.0), 3)
    return tensor_shifted

def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def pad_image(field, target_shape, pytorch=True, stacked_complex=True, padval=0, mode='constant'):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if pytorch:
        if stacked_complex:
            size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(target_shape) - np.array(field.shape[-2:])
        odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if stacked_complex:
                return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
            else:
                return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field


def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang


def pad_stacked_complex(field, pad_width, padval=0, mode='constant'):
    """Helper for pad_image() that pads a real padval in a complex-aware manner"""
    if padval == 0:
        pad_width = (0, 0, *pad_width)  # add 0 padding for stacked_complex dimension
        return nn.functional.pad(field, pad_width, mode=mode)
    else:
        if isinstance(padval, torch.Tensor):
            padval = padval.item()

        real, imag = field[..., 0], field[..., 1]
        real = nn.functional.pad(real, pad_width, mode=mode, value=padval)
        imag = nn.functional.pad(imag, pad_width, mode=mode, value=0)
        return torch.stack((real, imag), -1)

def crop_image(field, target_shape, pytorch=True, stacked_complex=True):
    """Crops a 2D field, see pad_image() for details

    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if pytorch:
        if stacked_complex:
            size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-3:-1]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2
    else:
        size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if pytorch and stacked_complex:
            return field[(..., *crop_slices, slice(None))]
        else:
            return field[(..., *crop_slices)]
    else:
        return field

def phasemap_8bit(phasemap, inverted=True):
    """convert a phasemap tensor into a numpy 8bit phasemap that can be directly displayed

    Input
    -----
    :param phasemap: input phasemap tensor, which is supposed to be in the range of [-pi, pi].
    :param inverted: a boolean value that indicates whether the phasemap is inverted.

    Output
    ------
    :return: output phasemap, with uint8 dtype (in [0, 255])
    """

    output_phase = ((phasemap + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    else:
        phase_out_8bit = ((output_phase) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)  # quantized to 8 bits
    return phase_out_8bit

def get_psnr_ssim(recon_amp, target_amp, multichannel=False):
    """get PSNR and SSIM metrics"""
    psnrs, ssims = {}, {}

    # amplitude
    target_linear = target_amp**2
    recon_linear = recon_amp**2

    target_srgb = srgb_lin2gamma(np.clip(target_linear, 0.0, 1.0))
    recon_srgb = srgb_lin2gamma(np.clip(recon_linear, 0.0, 1.0))
    psnrs = psnr(target_srgb, recon_srgb)
    ssims = ssim(target_srgb, recon_srgb, channel_axis=multichannel)

    return psnrs, ssims

def im2float(im, dtype=np.float32):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def resize_keep_aspect(image, target_res, pad=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = pad_image(image, resized_res, pytorch=False)
    else:
        image = crop_image(image, resized_res, pytorch=False)

    # switch to numpy channel dim convention, resize, switch back
    image = np.transpose(image, axes=(1, 2, 0))
    # image = resize(image, target_res, mode='reflect')  #'constant', 'edge', 'symmetric', 'reflect', 'wrap'
    image = cv2.resize(image, (target_res[1], target_res[0]), interpolation=cv2.INTER_CUBIC)
    image = np.reshape(image, (*target_res, -1))
    return np.transpose(image, axes=(2, 0, 1))


def replace_amplitude(field, amplitude):
    """takes a Complex tensor with real/imag channels, converts to
    amplitude/phase, replaces amplitude, then converts back to real/imag

    resolution of both Complex64 tensors should be (M, N, height, width)
    """
    # replace amplitude with target amplitude and convert back to real/imag
    real, imag = polar_to_rect(amplitude, field.angle())

    # concatenate
    return torch.complex(real, imag)

def srgb_gamma2lin(im_in):
    """converts from sRGB to linear color space"""
    thresh = 0.04045
    im_out = np.where(im_in <= thresh, im_in / 12.92, ((im_in + 0.055) / 1.055)**(2.4))
    return im_out

def srgb_lin2gamma(im_in):
    """converts from linear to sRGB color space"""
    thresh = 0.0031308
    im_out = np.where(im_in <= thresh, 12.92 * im_in, 1.055 * (im_in**(1 / 2.4)) - 0.055)
    return im_out

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    logging.basicConfig(filename=log_path, filemode="w")
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def log_write(logger,opt):
    logger.info("========================================")
    logger.info("run_id: %s" % opt.run_id)
    logger.info("num_epochs: %d" % opt.num_epochs)
    logger.info("size_of_miniBatches: %d" % opt.size_of_miniBatches)
    logger.info("lr: %f" % opt.lr)
    logger.info("save_pth: %s" % opt.save_pth)
    logger.info("device: %s" % opt.device)
    logger.info("layer_num: %d" % opt.layer_num)
    logger.info("distance_range: %f" % opt.distance_range)
    logger.info("img_distance: %f" % opt.img_distance)
    logger.info("dataset_average: %s" % opt.dataset_average)
    logger.info("img_size: %d" % opt.img_size)
    logger.info("data_path: %s" % opt.data_path)
    logger.info("base_channel: %d" % opt.base_channel)
    logger.info("num_layers: %d" % opt.num_layers)
    logger.info("num_filters_per_layer: %d" % opt.num_filters_per_layer)

    logger.info('cosineLR: {}'.format(opt.cosineLR))
    logger.info('cosineWarm: {}'.format(opt.cosineWarm))
    logger.info('stepLR: {}'.format(opt.stepLR))
    logger.info('stepLR_step_gamma: {}'.format(opt.stepLR_step_gamma))
    logger.info("p_loss: %s" % opt.p_loss)
    logger.info("f_loss: %s" % opt.f_loss)
    logger.info("ms_loss: %s" % opt.ms_loss)
    logger.info("l1_loss: %s" % opt.l1_loss)
    logger.info("l2_loss: %s" % opt.l2_loss)
    logger.info("p_loss_weight: %f" % opt.p_loss_weight)
    logger.info("f_loss_weight: %f" % opt.f_loss_weight)    
    logger.info("ms_loss_weight: %f" % opt.ms_loss_weight)
    logger.info("l1_loss_weight: %f" % opt.l1_loss_weight)
    logger.info("l2_loss_weight: %f" % opt.l2_loss_weight)
    logger.info("========================================")
    


#归一化
def normalize(x):
    x = x - x.min()
    x = x / x.max()
    return x