import os
import numpy as np
import logging

from dataset.data_util import *
logger = logging.getLogger(__package__)

# Resize all images of the scene to the same size
image_size_table ={
    "tat_intermediate_Playground": [548, 1008],
    "tat_intermediate_Family": [1084, 1957],
    "tat_intermediate_Francis": [1086, 1959],
    "tat_intermediate_Horse": [1084, 1958],
    "tat_training_Truck": [546, 980]
}

def load_data_split(basedir, scene, split, try_load_min_depth=True, only_img_files=False, seed=None):
    """
    :param split train | validation | test
    """
    scenes = sorted(os.listdir(basedir))
    all_ray_samplers = []
    
    scene_dir = os.path.join(basedir, scene, split)
    
    # camera parameters files
    intrinsics_files = find_files(os.path.join(scene_dir, "intrinsics"), exts=['*.txt'])
    pose_files = find_files(os.path.join(scene_dir, "pose"), exts=['*.txt'])
    img_files = find_files(os.path.join(scene_dir, "rgb"), exts=['*.png', '*.jpg'])
    
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))
    logger.info('raw img_files: {}'.format(len(img_files)))

    cam_cnt = len(pose_files)
    logger.info("Dataset len is {}".format(cam_cnt))
    assert(len(img_files) == cam_cnt)

    # img files
    style_dir = os.path.join("./wikiart", split)
    style_img_files = find_files(style_dir, exts=['*.png', '*.jpg'])
    logger.info("Number of style images is {}".format(len(style_img_files)))
    
    # create ray samplers
    ray_samplers = []
    H, W = image_size_table[scene]
    
    if seed != None:
        np.random.seed(seed)

    for i in range(cam_cnt):
        intrinsics = parse_txt(intrinsics_files[i])
        pose = parse_txt(pose_files[i])

        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                img_path = img_files[i],
                                                mask_path=None,
                                                min_depth_path=None,
                                                max_depth=None,
                                                style_imgs = style_img_files
                                                ))
        
    logger.info('Split {}, # views: {}'.format(split, cam_cnt))
    
    return ray_samplers


def load_data_split_scannet(basedir, scene, style_dir, mode="train", try_load_min_depth=True, only_img_files=False, seed=None):
    """
    :param split train | validation | test
    """

    scene_dir = os.path.join(basedir, scene)

    # camera parameters files
    intrinsic_file = find_files(scene_dir, exts=['*.txt'])
    pose_files = find_files(os.path.join(scene_dir, "pose"), exts=['*.txt'])
    img_files = find_files(os.path.join(scene_dir, "color"), exts=['*.png', '*.jpg'])

    logger.info('raw intrinsics_files: {}'.format(len(intrinsic_file)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))
    logger.info('raw img_files: {}'.format(len(img_files)))

    cam_cnt = len(pose_files)
    logger.info("Dataset len is {}".format(cam_cnt))
    assert (len(img_files) == cam_cnt)

    # img files
    style_img_files = find_files(style_dir, exts=['*.png', '*.jpg'])
    logger.info("Number of style images is {}".format(len(style_img_files)))

    # create ray samplers
    ray_samplers = []
    H, W = (240, 320)

    def get_intrinsics(scene_path):
        """
        Return 3x3 numpy array as intrinsic K matrix for the scene and (W,H) image dimensions if available
        """
        intrinsics = np.identity(4, dtype=np.float32)
        w = 0
        h = 0
        file = [os.path.join(scene_path, f) for f in os.listdir(scene_path) if ".txt" in f]
        if len(file) == 1:
            file = file[0]
            with open(file) as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    if "fx_color" in l:
                        fx = float(l.split(" = ")[1])
                        intrinsics[0,0] = fx
                    if "fy_color" in l:
                        fy = float(l.split(" = ")[1])
                        intrinsics[1,1] = fy
                    if "mx_color" in l:
                        mx = float(l.split(" = ")[1])
                        intrinsics[0,2] = mx
                    if "my_color" in l:
                        my = float(l.split(" = ")[1])
                        intrinsics[1,2] = my
                    if "colorWidth" in l:
                        w = int(l.split(" = ")[1])
                    if "colorHeight" in l:
                        h = int(l.split(" = ")[1])

        return intrinsics, (w,h)

    def modify_intrinsics_matrix(intrinsics, intrinsics_image_size, rgb_image_size):
        if intrinsics_image_size != rgb_image_size:
            intrinsics = np.array(intrinsics)
            intrinsics[0, 0] = (intrinsics[0, 0] / intrinsics_image_size[0]) * rgb_image_size[0]
            intrinsics[1, 1] = (intrinsics[1, 1] / intrinsics_image_size[1]) * rgb_image_size[1]
            intrinsics[0, 2] = (intrinsics[0, 2] / intrinsics_image_size[0]) * rgb_image_size[0]
            intrinsics[1, 2] = (intrinsics[1, 2] / intrinsics_image_size[1]) * rgb_image_size[1]

        return intrinsics

    intrinsics, size = get_intrinsics(scene_dir)
    intrinsics = modify_intrinsics_matrix(intrinsics, size, (W,H))

    if seed != None:
        np.random.seed(seed)

    indxs = [i for i in range(cam_cnt)]
    if mode == "train":
        indxs = np.delete(indxs, np.arange(0, len(indxs), 10))
    else:
        indxs = indxs[::10]

    for i in indxs:
        pose = parse_txt(pose_files[i])
        #print(pose)
        pose[:3, 3] *= 0.1
        #raise ValueError(pose)

        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                  img_path=img_files[i],
                                                  mask_path=None,
                                                  min_depth_path=None,
                                                  max_depth=None,
                                                  style_imgs=style_img_files
                                                  ))

    logger.info('Split {}, # views: {}'.format(scene, cam_cnt))

    return ray_samplers
