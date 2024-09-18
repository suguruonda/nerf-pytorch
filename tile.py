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

import glob

import cv2

base_path = "/home/ondas/projects/nerf-pytorch/logs2/f*"
f_list = glob.glob(base_path)
f_list.sort()
psnrs_all = []
ssims_all = []
lpipses_all = []
ini_glob = True
for j in [0,4,26,33,25]:
    path_ = f_list[j]
    index = int(os.path.basename(path_)[3:6])
    gt_images = load_fineview_data2('/home/ondas/nobackup/archive/360camera/butterfly/crop_undistort/', index, 8, True)

    test_image_path = path_ + "/testset_200000/*.png"
    imgs_list = glob.glob(test_image_path)
    imgs_list.sort()
    ini = True
    #breakpoint()
    for  i in [0,9,10,15,22,24,28]:
        out_img = imageio.imread(imgs_list[i])
        target_img = gt_images[i]*255
        H,W,_ = out_img.shape
        if H < W:
            C = H
            cut = (W-H)//2
            out_img = out_img[:,cut:cut+C,:]
            target_img = target_img[:,cut:cut+C,:]

        else:
            C = W
            cut = (H-W)//2
            out_img = out_img[cut:cut+C,:,:]
            target_img = target_img[cut:cut+C,:,:]
        out_img = cv2.resize(out_img,(100,100))
        target_img = cv2.resize(target_img,(100,100))
        up_down = np.vstack((target_img,out_img))
        
        if ini:
            row_left = up_down
            ini = False
        else:
            row_left = np.hstack((row_left, up_down))
    if ini_glob:
        out = row_left
        ini_glob = False
    else:
        out = np.vstack((out,row_left))

imageio.imwrite("/home/ondas/projects/nerf-pytorch/logs2/nerf_vs_gt.png",out.astype(np.uint8))

