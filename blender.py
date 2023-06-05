import torch
import json
import os
from torch.utils.data import Dataset
import numpy as np
import imageio 
from utils.logging import logger
from utils.ray_utils import get_ray_directions, get_rays

class BlenderDataSet(Dataset):
    def __init__(self, base_dir='./nerf_synthetic', scene='lego', mode='train'):
        super(BlenderDataSet, self).__init__()
        self.root_dir = os.path.join(base_dir, scene)
        self.mode = mode
        assert self.mode in ['train', 'test']
        logger.title("Loading dataset from {}, mode: {}".format(self.root_dir, self.mode))
        self.images = []
        self.poses = []
        self.read_transforms_json(os.path.join(self.root_dir, 'transforms_{}.json'.format(mode)))
        logger.info("rays shape: {}".format(self.rays.shape))
        logger.info("directions shape: {}".format(self.directions.shape))
        logger.info("images shape: {}".format(self.images.shape))
        logger.info("poses shape: {}".format(self.poses.shape))
        logger.info("intrinsic: {}".format(self.K))
        
    def read_transforms_json(self, path):
        with open(path, 'r') as fp:
            meta = json.load(fp)
        for frame in meta['frames']:
            img_path = os.path.join(self.root_dir, frame['file_path']) + '.png'
            self.images.append(np.array((imageio.v2.imread(img_path)) / 255.0).astype(np.float32))
            self.poses.append(np.array(frame['transform_matrix']).astype(np.float32))
        Height, Width, C = self.images[0].shape # (800, 800, 4) 
        self.h = Height
        self.w = Width
        assert C == 4 # 4 Channels RGBA
        self.images = torch.from_numpy(np.stack(self.images, 0))
        self.poses = torch.from_numpy(np.stack(self.poses, 0))
        # To get white background, we need to fuse alpha channel with rgb channel
        # without fusion and simply convert to RGB by indexing, we get black background: self.images = self.images[...,:3]
        self.images = self.images[...,:3] * self.images[...,-1:] + (1.0 - self.images[...,-1:]) # (N, H, W, 3)
        camera_angle_x = float(meta['camera_angle_x'])
        focal = 0.5 * Width / np.tan(0.5 * camera_angle_x)
        self.K = torch.from_numpy(np.array([
            [focal, 0, 0.5 * Width],
            [0, focal, 0.5 * Height],
            [0, 0, 1]
        ]))
        directions = get_ray_directions(Height, Width, focal) # (H, W, 3)
        self.rays = []
        self.directions = []
        for c2w_pose in self.poses:
            rays_o, rays_d = get_rays(directions, c2w_pose[:3, :]) # (H*W, 3), (H*W, 3)
            self.rays.append(rays_o)
            self.directions.append(rays_d) # normalized

        self.rays = torch.stack(self.rays) # (N, H*W, 3)
        self.directions = torch.stack(self.directions) # (N, H*W, 3)
        
        if self.mode == 'train':
            self.rays = self.rays.view(-1, 3) # (N*H*W, 3)
            self.directions = self.directions.view(-1, 3) # (N*H*W, 3)
            self.images = self.images.view(-1, 3) # (N*H*W, 3)
        else:
            self.images = self.images.view(-1, Height*Width, 3) # (N, H*W, 3)
         
    def __len__(self):
        return len(self.rays)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return {
                'rays_o': self.rays[idx], # (B, 3)
                'rays_d': self.directions[idx], # (B, 3)
                'rgbs': self.images[idx], # (B, 3)
            }
        else:
            return {
                'rays_o': self.rays[idx], # (B, H*W, 3)
                'rays_d': self.directions[idx], # (B, H*W, 3)
                'rgbs': self.images[idx], # (B, H*W, 3)
            }
        
    
from torch.utils.data import DataLoader
if __name__ == '__main__':
    dataset = BlenderDataSet()
    dataloader = DataLoader(dataset)
    
    # To save image:
    # i1 = self.images[0] * 255
    # imageio.v2.imwrite('test2.png', i1.astype(np.uint8))        