import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

#this is for perturbation
def rotation_matrix(theta_x,theta_y,theta_z):
    Rx = np.array([
        [1,0,0,0],
        [0,np.cos(theta_x),-np.sin(theta_x),0],
        [0,np.sin(theta_x), np.cos(theta_x),0],
        [0,0,0,1]], dtype='f')

    Ry = np.array([
        [np.cos(theta_y),0,-np.sin(theta_y),0],
        [0,1,0,0],
        [np.sin(theta_y),0, np.cos(theta_y),0],
        [0,0,0,1]], dtype='f')

    Rz = np.array([
        [np.cos(theta_x),-np.sin(theta_x),0,0],
        [np.sin(theta_x), np.cos(theta_x),0,0],
        [0,0,1,0],
        [0,0,0,1]], dtype='f')
    
    return Rz @ Ry @ Rx

def translation_matrix(theta_x,theta_y,theta_z):
    Rx = np.array([
        [1,0,0,theta_x],
        [0,1,0,theta_y],
        [0,0,1,theta_z],
        [0,0,0,1]], dtype='f')
    return Rz 

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, pose_perturbation=False, pprate=1e-0):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    #pose perturbation
    np.random.seed(0)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))

            if pose_perturbation == True:
                tx,ty,tz = pprate * (np.random.rand(3) - 0.5) / 180. * np.pi
                random_pose_angle = rotation_matrix(tx,ty,tz)
                random_pose_t = translation_matrix(np.random.rand(3) - 0.5) * 9
                pose_w2c = np.linalg.inv(np.array(frame['transform_matrix']))
                #add_perturbation = random_pose_angle @ pose_w2c
                add_perturbation = random_pose_angle @ pose_w2c + random_pose_t
                poses.append(np.linalg.inv(add_perturbation))
            else:
                poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split


