import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from load_fineview import load_fineview_data2
from skimage.metrics import structural_similarity        


import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg')
def lpips_fn(x, y):
    '''
        x: [H,W,3]
        y: [H,W,3]
    '''
    x = torch.from_numpy(x).float().permute(2,0,1).unsqueeze(0)
    y = torch.from_numpy(y).float().permute(2,0,1).unsqueeze(0)

    loss = loss_fn_vgg(x, y, normalize=True)
    loss = loss.item()
    return loss

import glob
base_path = "/home/ondas/projects/nerf-pytorch/logs2/f*"
f_list = glob.glob(base_path)
f_list.sort()
psnrs_all = []
ssims_all = []
lpipses_all = []
for i in f_list:
    print(i)
    index = int(os.path.basename(i)[3:6])
    gt_images = load_fineview_data2('/home/ondas/nobackup/archive/360camera/butterfly/crop_undistort/', index, 8, True)

    test_image_path = i + "/testset_200000/*.png"
    imgs_list = glob.glob(test_image_path)
    imgs_list.sort()
    psnrs = []
    ssims = []
    lpipses = []
    for count, ele in enumerate(gt_images):

        out_img = imageio.imread(imgs_list[count])
        out_img = (np.array(out_img) / 255.).astype(np.float32) 
        target_img = ele
        psnr = -10. * np.log10(np.mean(np.square(out_img - target_img)))
        ssim = structural_similarity(out_img, target_img, data_range=1.0, channel_axis = 2)
        
        lpips = lpips_fn(target_img, out_img)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipses.append(lpips)

    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    lpipses = np.array(lpipses)

    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    avg_lpips = np.mean(lpipses)


    psnrs_all.append(avg_psnr)
    ssims_all.append(avg_ssim)
    lpipses_all.append(avg_lpips)
    with open(i + "/testset_200000" + "/metrics.txt", mode="w", newline='\n') as f:
        f.write("Mean PSNR " + str(avg_psnr) +"\n")
        f.write("Mean SSIM " + str(avg_ssim) +"\n")
        f.write("Mean LPIPS " + str(avg_lpips) +"\n")
psnrs_all = np.array(psnrs_all)
ssims_all = np.array(ssims_all)
lpipses_all = np.array(lpipses_all)
avg_psnr_all = np.mean(psnrs_all)
avg_ssim_all = np.mean(ssims_all)
avg_lpips_all = np.mean(lpipses_all)
with open("/home/ondas/projects/nerf-pytorch/logs2" + "/all_metrics.txt", mode="w", newline='\n') as f:
    f.write("Mean PSNR " + str(avg_psnr_all) +"\n")
    f.write("Mean SSIM " + str(avg_ssim_all) +"\n")
    f.write("Mean LPIPS " + str(avg_lpips_all) +"\n")
with open("/home/ondas/projects/nerf-pytorch/logs2" + "/all_metrics_psnr.txt", mode="w", newline='\n') as f:
    for count,ele in enumerate(psnrs_all):
        f.write(f_list[count] + " " + str(ele) +"\n")
with open("/home/ondas/projects/nerf-pytorch/logs2" + "/all_metrics_ssim.txt", mode="w", newline='\n') as f:
    for count,ele in enumerate(ssims_all):
        f.write(f_list[count] + " " + str(ele) +"\n")
with open("/home/ondas/projects/nerf-pytorch/logs2" + "/all_metrics_lpips.txt", mode="w", newline='\n') as f:
    for count,ele in enumerate(lpipses_all):
        f.write(f_list[count] + " " + str(ele) +"\n")
