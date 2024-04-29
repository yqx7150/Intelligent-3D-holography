import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import configargparse
from tensorboardX import SummaryWriter
import math
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)

# dataset/model/loss function
from dataLoader import data_loader
import dataLoader
from rtholo import rtholo
import perceptualloss as perceptualloss  # perceptual loss
from focal_frequency_loss import FocalFrequencyLoss as FFL  # focal frequency loss
from pytorch_msssim import SSIM, MS_SSIM  # ms-ssim loss
import cv2 as cv
import shutil
import logging
import numpy as np
from utils import *
from rich.progress import track

import shutil

# Command line argument processing
p = configargparse.ArgumentParser()
p.add_argument("-c","--config_filepath",required=False,is_config_file=True,help="Path to config file.")
p.add_argument("--run_id", type=str, default="CNN_test", help="Experiment name", required=False)  # 命令行
p.add_argument("--num_epochs", type=int, default=90, help="Number of epochs")
p.add_argument("--size_of_miniBatches", type=int, default=1, help="Size of minibatch")
p.add_argument("--lr", type=float, default=1e-3, help="learning rate of Holonet weights")
p.add_argument("--save_pth", type=str, default="../save/", help="Path to data directory")
p.add_argument("--device", type=str, default="0", help="Path to data directory")  # 命令行
p.add_argument("--layer_num", type=int, default=30, help="Number of layers")
p.add_argument("--distance_range", type=float, default=0.03, help="Distance range")
p.add_argument("--img_distance", type=float, default=0.2, help="Distance range")
p.add_argument("--dataset_average", action="store_true", help="Dataset_average")
p.add_argument("--log_path", type=str, default="../log/", help="Path to data directory")
p.add_argument("--img_size", type=int, default=1024, help="Size of image")  # 命令行
p.add_argument("--data_path",type=str,default="../../mit-4k",help="Path to data directory",)
p.add_argument("--feature_size", type=float, default=7.48e-6, help="base channel of U-Net")  # 命令行
p.add_argument("--num_layers", type=int, default=10, help="Number of layers")  # 命令行
p.add_argument("--num_filters_per_layer", type=int, default=15, help="Number of filters per layer")  # 命令行

p.add_argument("--cosineLR", action="store_true", help="Use cosine learning rate")
p.add_argument("--cosineWarm", action="store_true", help="Use cosine warm learning rate")
p.add_argument("--stepLR", action="store_true", help="Use step learning rate")
p.add_argument("--stepLR_step_size", type=int, default=1, help="stepLR step size")
p.add_argument("--stepLR_step_gamma", type=float, default=0.8, help="stepLR step gamma")
p.add_argument("--CNNPP", action="store_true", help="Use CNNPP")


#!损失函数选择
p.add_argument("--p_loss", action="store_true", help="Use perceptual loss")
p.add_argument("--f_loss", action="store_true", help="Use focal frequency loss")
p.add_argument("--ms_loss", action="store_true", help="Use ms_ssim loss")
p.add_argument("--l1_loss", action="store_true", help="Use L1 loss")
p.add_argument("--l2_loss", action="store_true", help="Use L2 loss")

#!损失函数权重
p.add_argument("--p_loss_weight", type=float, default=1.0, help="perceptual loss weight")
p.add_argument("--f_loss_weight", type=float, default=1.0, help="focal frequency loss weight")
p.add_argument("--ms_loss_weight", type=float, default=1.0, help="ms_ssim loss weight")
p.add_argument("--l1_loss_weight", type=float, default=1.0, help="L1 loss weight")
p.add_argument("--l2_loss_weight", type=float, default=1.0, help="L2 loss weight")

p.add_argument("--clear", action="store_true", help="clear")
p.add_argument("--ckpt_continue", type=str, default=None, help="ckpt_continue")


# parse arguments
opt = p.parse_args()
run_id = opt.run_id

logger = logger_config(log_path=os.path.join(opt.log_path, run_id + ".log"))
# logger 打印所有参数
log_write(logger, opt)


os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

if not os.path.exists(os.path.join(opt.save_pth, run_id, "holo")):
    os.makedirs(os.path.join(opt.save_pth, run_id, "holo"))
else:
    if(opt.clear):
        shutil.rmtree(os.path.join(opt.save_pth, run_id, "holo"))
        os.makedirs(os.path.join(opt.save_pth, run_id, "holo"))
if not os.path.exists(os.path.join(opt.save_pth, run_id, "out_amp_mask")):
    os.makedirs(os.path.join(opt.save_pth, run_id, "out_amp_mask"))
else:
    if(opt.clear):
        shutil.rmtree(os.path.join(opt.save_pth, run_id, "out_amp_mask"))
        os.makedirs(os.path.join(opt.save_pth, run_id, "out_amp_mask"))
if not os.path.exists(os.path.join(opt.save_pth, run_id, "out_amp")):
    os.makedirs(os.path.join(opt.save_pth, run_id, "out_amp"))
else:
    if(opt.clear):
        shutil.rmtree(os.path.join(opt.save_pth, run_id, "out_amp"))
        os.makedirs(os.path.join(opt.save_pth, run_id, "out_amp"))
if not os.path.exists(os.path.join(opt.save_pth, run_id, "depth")):
    os.makedirs(os.path.join(opt.save_pth, run_id, "depth"))
else:
    if(opt.clear):
        shutil.rmtree(os.path.join(opt.save_pth, run_id, "depth"))
        os.makedirs(os.path.join(opt.save_pth, run_id, "depth"))
if not os.path.exists(os.path.join(opt.save_pth, run_id, "amp")):
    os.makedirs(os.path.join(opt.save_pth, run_id, "amp"))
else:
    if(opt.clear):
        shutil.rmtree(os.path.join(opt.save_pth, run_id, "amp"))
        os.makedirs(os.path.join(opt.save_pth, run_id, "amp"))

if not os.path.exists(os.path.join("checkpoints", run_id)):
    os.makedirs(os.path.join("checkpoints", run_id))


# tensorboard setup and file naming
time_str = str(datetime.now()).replace(" ", "-").replace(":", "-")
writer = SummaryWriter(f"runs/{run_id}")

device = torch.device("cuda")

# Image data for training
train_loader = data_loader(opt, type="train")
# val_loader = data_loader(opt, type="val")

# Load models #
self_holo = rtholo(
    size=opt.img_size,
    feature_size=opt.feature_size,
    distance_range=opt.distance_range,
    img_distance=opt.img_distance,
    layers_num=opt.layer_num,
    num_filters_per_layer=opt.num_filters_per_layer,
    num_layers=opt.num_layers,
    CNNPP=opt.CNNPP,
).to(device)

if(opt.ckpt_continue is not None):
    self_holo.load_state_dict(torch.load(opt.ckpt_continue))

self_holo.train()  # generator to be trained

# Loss function
if opt.p_loss:
    p_loss = perceptualloss.PerceptualLoss(lambda_feat=0.025)
    p_loss = p_loss.to(device)
if opt.f_loss:
    f_loss = FFL(loss_weight=1.0, alpha=1.0, patch_factor=1)
    f_loss = f_loss.to(device)
if opt.ms_loss:
    ms_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3)
    ms_loss = ms_loss.to(device)
if opt.l1_loss:
    l1_loss = nn.L1Loss()
    l1_loss = l1_loss.to(device)
if opt.l2_loss:
    l2_loss = nn.MSELoss()
    l2_loss = l2_loss.to(device)


mseloss = nn.MSELoss()
mseloss = mseloss.to(device)
# create optimizer
optvars = self_holo.parameters()
optimizer = optim.Adam(optvars, lr=opt.lr)

if opt.cosineLR is True:
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
elif opt.cosineWarm is True:
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=0, last_epoch=-1
    )
elif opt.stepLR is True:
    lr_scheduler = StepLR(
        optimizer, step_size=opt.stepLR_step_size, gamma=opt.stepLR_step_gamma
    )
else:
    lr_scheduler = None

layer_weight = np.zeros(opt.layer_num)
ikk_probability = np.zeros(opt.layer_num)


# Training loop #
for i in range(opt.num_epochs):
    train_loss = []
    # self_holo.train()  # generator to be trained
    dataLoader.epoch_num = i
    for k, target in enumerate(train_loader):
        # get target image
        amp, depth, mask, ikk = target
        ikk_probability[ikk] += 1
        amp, depth, mask = amp.to(device), depth.to(device), mask.to(device)
        source = torch.cat([amp, depth], dim=-3)  # 连接振幅和深度

        optimizer.zero_grad()

        ik = k + i * len(train_loader)

        holo, slm_amp, recon_field = self_holo(source, ikk)

        output_amp = 0.95 * recon_field.abs()
        output_amp_save = output_amp
        output_amp = output_amp * mask
        save = amp * mask

        output_amp = output_amp.repeat(1, 3, 1, 1)
        amp_i = amp * mask
        amp_i = amp_i.repeat(1, 3, 1, 1)

        if ik % 100 == 0:
            cv.imwrite(
                os.path.join(opt.save_pth, run_id, "out_amp", str(ik) + ".png"),
                normalize(output_amp_save[0, 0, ...].detach().cpu().numpy()) * 255,
            )
            cv.imwrite(
                os.path.join(opt.save_pth, run_id, "depth", str(ik) + ".png"),
                normalize(depth[0, 0, ...].detach().cpu().numpy()) * 255,
            )
            cv.imwrite(
                os.path.join(opt.save_pth, run_id, "amp", str(ik) + ".png"),
                normalize(amp[0, 0, ...].detach().cpu().numpy()) * 255,
            )
            cv.imwrite(
                os.path.join(opt.save_pth, run_id, "out_amp_mask", str(ik) + ".png"),
                normalize(save[0, 0, ...].detach().cpu().numpy()) * 255,
            )
            cv.imwrite(
                os.path.join(opt.save_pth, run_id, "holo", str(ik) + ".png"),
                normalize(holo[0, 0, ...].detach().cpu().numpy()) * 255,
            )

        # loss
        loss_val = 0
        if opt.p_loss:
            loss_val += p_loss(output_amp, amp_i) * opt.p_loss_weight
        if opt.f_loss:
            loss_val += f_loss(output_amp, amp_i) * opt.f_loss_weight
        if opt.ms_loss:
            loss_val += (1 - ms_loss(output_amp, amp_i)) * opt.ms_loss_weight
        if opt.l1_loss:
            loss_val += l1_loss(output_amp, amp_i) * opt.l1_loss_weight
        if opt.l2_loss:
            loss_val += l2_loss(output_amp, amp_i) * opt.l2_loss_weight

        loss_val += 0.1 * mseloss(slm_amp.mean(), slm_amp)   #!

        train_loss.append(loss_val.item())

        loss_val.backward()
        optimizer.step()

        # print and output to tensorboard
        # print(f'iteration {ik}: {loss_val.item()}')
        # logger.info(f'epoch {i}, iteration {k},ikk {ikk.item()}: {loss_val.item()}')
        distance = (0 - opt.distance_range) / opt.layer_num * ikk
        dis = distance - opt.img_distance
        logger.info(
            "epoch:%02d || iteration:%04d || ikk:%03d || dis: %.6f|| loss:%.6f || lr:%.8f"
            % (
                i,
                k,
                ikk.item(),
                dis,
                loss_val.item(),
                optimizer.state_dict()["param_groups"][0]["lr"],
            )
        )

        with torch.no_grad():
            # writer.add_scalar('train_Loss', loss_val, ik)

            if ik % 100 == 0:
                writer.add_scalar("train_Loss", np.mean(train_loss), ik)
                train_loss = []
                writer.add_image("amp", (amp[0, ...]), ik)
                writer.add_image("depth", (depth[0, ...]), ik)
                writer.add_image("output_amp", (output_amp[0, ...]), ik)
                # normalize SLM phase
                writer.add_image(
                    "SLM_Phase", (holo[0, ...] + math.pi) / (2 * math.pi), ik
                )

    if lr_scheduler is not None:
        lr_scheduler.step()

    # writer.add_scalar('train_loss', np.mean(train_loss), i)

    # save trained model
    torch.save(
        self_holo.state_dict(),
        os.path.join("checkpoints", run_id, "{}.pth".format(i + 1)),
    )

    for j in range(opt.layer_num):
        layer_weight[j] = ikk_probability[j] / np.sum(ikk_probability)
        logger.info("layer({:02d}):{:.4f}".format(j, layer_weight[j]))
