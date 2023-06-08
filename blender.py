import torch
import json
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import imageio 
from utils.log import logger
from utils.ray_utils import get_ray_directions, get_rays

class BlenderDataSet(Dataset):
    def __init__(self, base_dir='./nerf_synthetic', scene='lego', mode='train', inference_train=False):
        super(BlenderDataSet, self).__init__()
        self.root_dir = os.path.join(base_dir, scene)
        self.mode = mode
        assert self.mode in ['train', 'test']
        logger.title("Loading dataset from {}, mode: {}".format(self.root_dir, self.mode))
        self.images = []
        self.poses = []
        if inference_train == True:
            mode = 'train'
            logger.info("inference train images")
        self.read_transforms_json(os.path.join(self.root_dir, 'transforms_{}.json'.format(mode)))
        logger.info("rays shape: {}".format(self.rays.shape))
        logger.info("directions shape: {}".format(self.directions.shape))
        logger.info("images shape: {}".format(self.images.shape))
        logger.info("poses shape: {}".format(self.poses.shape))
        logger.info("intrinsic: {}".format(self.K))
        
    def read_transforms_json(self, path):
        with open(path, 'r') as fp:
            meta = json.load(fp)
        for frame in tqdm(meta['frames'], desc='Loading image frames', leave=True):
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
        self.raw_directions = get_ray_directions(Height, Width, focal) # (H, W, 3)
        self.rays = []
        self.directions = []
        for c2w_pose in self.poses:
            rays_o, rays_d = get_rays(self.raw_directions, c2w_pose[:3, :]) # (H*W, 3), (H*W, 3)
            self.rays.append(rays_o)
            self.directions.append(rays_d) # normalized

        self.rays = torch.stack(self.rays) # (N, H*W, 3)
        self.directions = torch.stack(self.directions) # (N, H*W, 3)
        
        self.bounding_box = self.get_bbox3d_for_blenderobj(near=2, far=6)
        
        if self.mode == 'train':
            self.rays = self.rays.view(-1, 3) # (N*H*W, 3)
            self.directions = self.directions.view(-1, 3) # (N*H*W, 3)
            self.images = self.images.view(-1, 3) # (N*H*W, 3)
        else:
            self.images = self.images.view(-1, Height*Width, 3) # (N, H*W, 3)
            
    def get_bbox3d_for_blenderobj(self, near, far):
        min_bound = [100, 100, 100]
        max_bound = [-100, -100, -100]
        
        for rays_o, rays_d in zip(self.rays, self.directions):
            
            def find_min_max(pt):
                for i in range(3):
                    if(min_bound[i] > pt[i]):
                        min_bound[i] = pt[i]
                    if(max_bound[i] < pt[i]):
                        max_bound[i] = pt[i]
                return

            for i in [0, self.w-1, self.h*self.w-self.w, self.h*self.w-1]:
                min_point = rays_o[i] + near*rays_d[i]
                max_point = rays_o[i] + far*rays_d[i]
                find_min_max(min_point)
                find_min_max(max_point)

        min_bound = torch.tensor(min_bound) - torch.tensor([1.0,1.0,1.0])
        max_bound = torch.tensor(max_bound) + torch.tensor([1.0,1.0,1.0])
        return (min_bound, max_bound)
            
         
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
    print(dataset.min_bound, dataset.max_bound)
    # dataloader = DataLoader(dataset)
    
    # To save image:
    # i1 = self.images[0] * 255
    # imageio.v2.imwrite('test2.png', i1.astype(np.uint8))        