import os
from pathlib import Path
#import torch
import numpy as np
import imageio 
#import json
#import torch.nn.functional as F
import cv2
import h5py
import glob
import open3d as o3d
########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            #return imageio.imread(f, ignoregamma=True)
            return imageio.imread(f)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []

    for th in np.linspace(0.,2.*np.pi, 120):

        #camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        #camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh + radcircle * np.sin(th)*0.3])
        camorigin = np.array([radcircle * np.cos(th), 0, zh + radcircle * np.sin(th)])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)
    """
    for th in np.linspace(0.,2.*np.pi, 360):

        #camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        #camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh + radcircle * np.sin(th)*0.3])
        camorigin = np.array([radcircle * np.cos(th*6), radcircle * np.sin(th*6), zh - radcircle * np.cos(th)])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)
    """
    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def cal_bds(xyz, poses):
  
    zvals = np.sum(-(xyz[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    print( 'Depth stats', zvals.min(), zvals.max(), zvals.mean() )
    
    bds = []
    for i in range(poses.shape[2]):
        zs = zvals[:, i]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )
        
        bds.append(np.array([close_depth, inf_depth]))
    return np.array(bds).T


def load_fineview_data(basedir, img_number, factor, crop, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):

    non_debug = True
    crop_param_path = basedir + '/crop_pram_undistort.h5'

    fc = h5py.File(crop_param_path, 'r')
    x_min,y_min = fc["crop/offset"][img_number]
    crop_image_size = fc["crop/img_size"][img_number]
    x_max = crop_image_size[0] + x_min
    y_max = crop_image_size[1] + y_min
    fc.close()

    c_param_path = basedir + '/camera_pram_2_no180_2_opt.h5'
    f = h5py.File(c_param_path, 'r')
  
    if crop:
        extention = 'png'
        img_folder ="crop_undistort"
    else:
        extention = 'JPG'
        img_folder ="undistort"

    folder_list = glob.glob(basedir + "/" + img_folder + "/*")
    folder_list.sort()

    l1 = glob.glob(folder_list[img_number] + '/images/camera1/*.' + extention)
    l2 = glob.glob(folder_list[img_number] + '/images/camera2/*.' + extention)
    l3 = glob.glob(folder_list[img_number] + '/images/camera3/*.' + extention)
    l4 = glob.glob(folder_list[img_number] + '/images/camera4/*.' + extention)
    l5 = glob.glob(folder_list[img_number] + '/images/camera5/*.' + extention)
    l6 = glob.glob(folder_list[img_number] + '/images/camera6/*.' + extention)
    l7 = glob.glob(folder_list[img_number] + '/images/camera7/*.' + extention)
    l8 = glob.glob(folder_list[img_number] + '/images/camera8/*.' + extention)
    l1.sort()
    l2.sort()
    l3.sort()
    l4.sort()
    l5.sort()
    l6.sort()
    l7.sort()
    l8.sort()

    #img_list = l1[::4] + l2[::2] +  l3 + l4 + l5 + l6 + l7[::2] + l8[::4]
    img_list = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
    sp_folder = os.path.basename(folder_list[img_number])

    img_list.sort()
    imgs = []
    imgs_mask = []
    w2c_mats = []
    testskip = 1
    if testskip==0:
        skip = 1
    else:
        skip = testskip
    
    if crop:    
        H_original = crop_image_size[1]
        W_original = crop_image_size[0]
    else:
        H_original, W_original = imageio.imread(img_list[0]).shape[:2]
    
    d_type = imageio.imread(img_list[0]).dtype
    fx = []
    fy = []
    K = []
    for i in ['camera1','camera2','camera3','camera4','camera5','camera6','camera7','camera8']:
        fx.append(f[i]['mtx'][0,0])
        fy.append(f[i]['mtx'][1,1])
    focals = fx + fy
    focal = np.array(focals).mean() 

    H = H_original//factor
    W = W_original//factor
    focal = focal/factor

    hwf = np.array([H,W,focal]).reshape([3,1])
    for i in img_list[::skip]:
        
        i_path = Path(i)
        mask_path = Path(basedir).joinpath('crop_mask_undistort', i_path.parts[-4], i_path.parts[-2], i_path.stem + "_mask.png")
        if non_debug:
            image_original = imageio.imread(i)
            
            if crop:
                image_blank = imageio.imread(mask_path)[:,:,0]
            else:
                image_blank = np.zeros((H_original,W_original), dtype=d_type)
                image_blank[y_min:y_max,x_min:x_max] = imageio.imread(mask_path)[:,:,0]
        else:
            image_original = np.empty((100,100,3))
            image_blank = np.empty((100,100,1))
    
        imgs.append(image_original)
        imgs_mask.append(image_blank)
        
        #pose conversion from fineview data
        camera = i[-14:-7]
        i_number = int(i[-6:-4])
        r_vec = f[camera]['rvec'][i_number]
        t_vec = f[camera]['tvec'][i_number]

        k_param = f[camera]['mtx'][:]
        k_param[0,2] -= x_min
        k_param[1,2] -= y_min
        K.append(k_param)

        mat = np.concatenate([r_vec, t_vec],  axis=1)
        tmp = np.array([0,0,0,1])
        mat4 = np.vstack((mat, tmp.T))
        w2c_mats.append(mat4)

    f.close()

    K = np.stack(K)
    K = K/factor
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)

    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)   
    #fineview pose is world to camera pose and it is same with opencv coordinate. Convert from (right, down, forward) to (right, up, backward) and change to camera to world coordinate 
    #must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    pc_path = basedir + "/correspondence/" + sp_folder + "/" + sp_folder + ".pcd"  
    pcd = o3d.io.read_point_cloud(pc_path)
    xyz = np.asarray(pcd.points)
    bds = cal_bds(xyz, poses)

    #render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    imgs_mask = (np.array(imgs_mask) / 255.).astype(np.float32) 

    if factor != 1:
        imgs_resized = np.zeros((imgs.shape[0], H, W, 4))
        for i in range(imgs.shape[0]):
            imgs_resized[i][:,:,0:3] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
            imgs_resized[i][:,:,3] = cv2.resize(imgs_mask[i], (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_resized
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    #no need this 
    #imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    
    #pose visualization
    """
    namename = "fb1"
    from camera_visualization import camera_visualizer
    visualizer = camera_visualizer()    
    outposes = render_poses[:,:3,:4]
    n_poses = np.zeros((outposes.shape[0],4,4))
    for count, i in enumerate(outposes[::1,:,:]):
        tmp = np.array([0,0,0,1])
        mat4 = np.vstack((i, tmp.T))
        mat_i = np.linalg.inv(mat4)
        n_poses[count] = mat4
    visualizer.plot_camera_scene(n_poses,0.2,"red","pose")
    visualizer.save("test_rendered_new_spherify" + namename + ".png")
    #visualizer.show()
    """
    """
    visualizer2 = camera_visualizer()    
    outposes2 = poses[:,:3,:4]
    n_poses2 = np.zeros((outposes2.shape[0],4,4))
    for count, i in enumerate(outposes2[::1,:,:]):
        tmp = np.array([0,0,0,1])
        mat4 = np.vstack((i, tmp.T))
        mat_i = np.linalg.inv(mat4)
        n_poses2[count] = mat4
    visualizer2.plot_camera_scene(n_poses2,0.2,"red","pose")
    visualizer2.save("test_poses_" + namename + ".png")
    #visualizer2.show()
    #breakpoint()
    """
    return images, poses, bds, render_poses, i_test, K

def load_fineview_data2(basedir, img_number, factor, crop, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):

    #breakpoint()
    non_debug = True

    if crop:
        extention = 'png'
        img_folder ="crop_undistort"
    else:
        extention = 'JPG'
        img_folder ="undistort"

    folder_list = glob.glob(basedir + "/" + img_folder + "/*")
    folder_list.sort()
    
    if crop:
        extention = 'png'
    else:
        extention = 'JPG'

    l1 = glob.glob(folder_list[img_number] + '/images/camera1/*.' + extention)
    l2 = glob.glob(folder_list[img_number] + '/images/camera2/*.' + extention)
    l3 = glob.glob(folder_list[img_number] + '/images/camera3/*.' + extention)
    l4 = glob.glob(folder_list[img_number] + '/images/camera4/*.' + extention)
    l5 = glob.glob(folder_list[img_number] + '/images/camera5/*.' + extention)
    l6 = glob.glob(folder_list[img_number] + '/images/camera6/*.' + extention)
    l7 = glob.glob(folder_list[img_number] + '/images/camera7/*.' + extention)
    l8 = glob.glob(folder_list[img_number] + '/images/camera8/*.' + extention)
    l1.sort()
    l2.sort()
    l3.sort()
    l4.sort()
    l5.sort()
    l6.sort()
    l7.sort()
    l8.sort()

    #img_list = l1[::4] + l2[::2] +  l3 + l4 + l5 + l6 + l7[::2] + l8[::4]
    img_list = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
    #img_list = l1[0:2]
    sp_folder = os.path.basename(folder_list[img_number])

    img_list.sort()
    #print(img_list)    
    imgs = []
    testskip = 1
    if testskip==0:
        skip = 1
    else:
        skip = testskip
    
    H_original, W_original = imageio.imread(img_list[0]).shape[:2]
    
    H = H_original//factor
    W = W_original//factor

    for i in img_list[::10]:
        
        i_path = Path(i)
        image_original = imageio.imread(i)
        imgs.append(image_original)
    imgs = (np.array(imgs)).astype(np.float32)
    if factor != 1:
        imgs_resized = np.zeros((imgs.shape[0], H, W, 3))
        for i in range(imgs.shape[0]):
            imgs_resized[i][:,:,0:3] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_resized
    imgs = (np.array(imgs) / 255.).astype(np.float32) 
    return imgs
