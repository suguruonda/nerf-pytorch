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

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from load_fineview import load_fineview_data
#from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer
#visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [-10, 10])
#visualizer2 = CameraPoseVisualizer([-7, 7], [-7, 7], [-7, 7])
#visualizer2 = CameraPoseVisualizer([-1, 1], [-1, 1], [-1, 1])
#visualizer = CameraPoseVisualizer([-0.5, 0.5], [-0.5, 0.5], [-3.75, -4.25])
from camera_visualization import camera_visualizer
visualizer = camera_visualizer()
#images, poses, render_poses, hwf, i_split = load_blender_data('./data/nerf_synthetic/lego', True, 8)
breakpoint()
images, poses, bds, render_poses, i_test = load_llff_data('./data/nerf_llff_data/b001_up', 8,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=False)

#images, poses, render_poses, hwf = load_fineview_data('/data2/suguru/datasets/360camera/butterfly/crop_undistort', 0, 8, 8)
#images, poses, bds, render_poses, i_test = load_fineview_data('/data2/suguru/datasets/360camera/butterfly/undistort', 0, 8, True, .75, False, False)
flag_fox = False
if flag_fox:
    with open('/mv_users/suguru/project/instant-ngp/data/nerf/fox/transforms.json', 'r') as fp:
        f = json.load(fp)
"""
for j in f["frames"]:
    i = np.array(j["transform_matrix"])
"""
#llff only
#breakpoint()
poses = poses[:,:3,:4]

n_poses = np.zeros((poses.shape[0],4,4))
for count, i in enumerate(poses):
    """
    r = i[0:3,0:3]
    t = i[0:3,3]
    r_n = r.T
    t_n = -r.T @ t
    mat = np.concatenate([r_n, t_n.reshape(3,1)],  axis=1)
    
    tmp = np.array([0,0,0,1])
    mat4 = np.vstack((mat, tmp.T))
    mat_i = np.linalg.inv(i)
    """
    #tmp = np.array([0,0,0,1])
    #i = np.vstack((i, tmp.T))
    #n_poses[count] = i
    #visualizer.extrinsic2pyramid(mat4, "red", 0.2)
    #x_180 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    """
    mat_i = np.linalg.inv(i)
    mat_i = np.concatenate([mat_i[0:1, :], -mat_i[1:2, :], -mat_i[2:3, :], mat_i[3:4, :]], 0)
    mat_i = np.linalg.inv(mat_i)
    """
    tmp = np.array([0,0,0,1])
    mat4 = np.vstack((i, tmp.T))
    mat_i = np.linalg.inv(mat4)

    n_poses[count] = mat4

    #visualizer2.extrinsic2pyramid(i, "red", 1)
#visualizer.plot_camera_scene(poses,0.5,"red","pose")
visualizer.plot_camera_scene(n_poses,0.5,"red","pose")
visualizer.save("test_b001_up_w2c.png")
#visualizer.show()
#visualizer2.show()
