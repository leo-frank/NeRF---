import torch
import json
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import imageio 
from easydict import EasyDict as edict
from utils.log import logger
from utils.ray_utils import get_ray_directions, get_rays

class BlenderDataSet(Dataset):
    def __init__(self, base_dir='./nerf_synthetic', scene='lego', mode='train', inference_train=False, noise = True):
        super(BlenderDataSet, self).__init__()
        self.root_dir = os.path.join(base_dir, scene)
        self.mode = mode
        self.noise = noise
        self.noise_level = 0.15     # This is a recommended level by BARF
        assert self.mode in ['train', 'test']
        logger.title("Loading dataset from {}, mode: {}".format(self.root_dir, self.mode))
        self.images = []
        self.raw_poses = []
        if inference_train == True: # read train json file but convert it to input as test mode
            mode = 'train'
            logger.info("inference train images")
        self.read_transforms_json(os.path.join(self.root_dir, 'transforms_{}.json'.format(mode)))
        logger.info("rays shape: {}".format(self.rays.shape))
        logger.info("directions shape: {}".format(self.directions.shape))
        logger.info("images shape: {}".format(self.images.shape))
        logger.info("poses shape: {}".format(self.raw_poses.shape))
        logger.info("intrinsic: {}".format(self.K))
        
    def read_transforms_json(self, path):
        with open(path, 'r') as fp:
            meta = json.load(fp)
        for frame in tqdm(meta['frames'], desc='Loading image frames', leave=True):
            img_path = os.path.join(self.root_dir, frame['file_path']) + '.png'
            self.images.append(np.array((imageio.v2.imread(img_path)) / 255.0).astype(np.float32))
            self.raw_poses.append(np.array(frame['transform_matrix']).astype(np.float32))
        Height, Width, C = self.images[0].shape # (800, 800, 4) 
        self.h = Height
        self.w = Width
        assert C == 4 # 4 Channels RGBA
        self.images = torch.from_numpy(np.stack(self.images, 0))
        self.raw_poses = torch.from_numpy(np.stack(self.raw_poses, 0))
        
        def taylor_A(x,nth=10):
            # Taylor expansion of sin(x)/x
            ans = torch.zeros_like(x)
            denom = 1.
            for i in range(nth+1):
                if i>0: denom *= (2*i)*(2*i+1)
                ans = ans+(-1)**i*x**(2*i)/denom
            return ans
        def taylor_B(x,nth=10):
            # Taylor expansion of (1-cos(x))/x**2
            ans = torch.zeros_like(x)
            denom = 1.
            for i in range(nth+1):
                denom *= (2*i+1)*(2*i+2)
                ans = ans+(-1)**i*x**(2*i)/denom
            return ans
        def taylor_C(x,nth=10):
            # Taylor expansion of (x-sin(x))/x**3
            ans = torch.zeros_like(x)
            denom = 1.
            for i in range(nth+1):
                denom *= (2*i+2)*(2*i+3)
                ans = ans+(-1)**i*x**(2*i)/denom
            return ans

        def skew_symmetric(w):
            w0,w1,w2 = w.unbind(dim=-1)
            O = torch.zeros_like(w0)
            wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                                torch.stack([w2,O,-w0],dim=-1),
                                torch.stack([-w1,w0,O],dim=-1)],dim=-2)
            return wx

        def se3_to_SE3(wu): # [...,3]
            w,u = wu.split([3,3],dim=-1)
            wx = skew_symmetric(w)
            theta = w.norm(dim=-1)[...,None,None]
            I = torch.eye(3,device=w.device,dtype=torch.float32)
            A = taylor_A(theta)
            B = taylor_B(theta)
            C = taylor_C(theta)
            R = I+A*wx+B*wx@wx
            V = I+B*wx+C*wx@wx
            Rt = torch.cat([R,(V@u[...,None])],dim=-1)
            return Rt

        def compose_pair(pose_a, pose_b):
            """
            Inputs:
                pose_a: (N, 3, 4)
                pose_b: (N, 3, 4)
            Outputs:
                pose_b: (N, 4, 4)
            """
            # pose_new(x) = pose_b o pose_a(x)
            R_a, t_a = pose_a[...,:3], pose_a[...,3:]
            R_b, t_b = pose_b[...,:3], pose_b[...,3:]
            R_new = R_b@R_a
            t_new = (R_b@t_a+t_b)[...,0]
            R_new = R_new.float()
            t_new = t_new.float()
            pose_new = torch.cat([R_new,t_new[...,None]],dim=-1) # [..., 3, 4]
            pose_bottom = torch.ones((pose_new.shape[0], 1, 4))
            pose_4x4 = torch.cat([pose_new, pose_bottom], dim = 1)
            return pose_4x4

        if self.noise:
            se3_noise = torch.randn(len(self.raw_poses), 6) * self.noise_level
            pose_noise = se3_to_SE3(se3_noise)
            self.noise_poses = compose_pair(pose_noise, self.raw_poses[:, :3, :]) # (N, 3, 4)
            self.poses = self.noise_poses
        else:
            self.poses = self.raw_poses
        
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
def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom
def invert(pose, use_inverse=False):
    # invert a camera pose
    R,t = pose[...,:3],pose[...,3:]
    R_inv = R.inverse()
    t_inv = (-R_inv@t)[...,0]
    pose_inv = self(R=R_inv,t=t_inv)
    return pose_inv

# basic operations of transforming 3D points between world/camera/image coordinates
def world2cam(X,pose): # [B,N,3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1,-2)

def cam2img(X,cam_intr):
    return X@cam_intr.transpose(-1,-2)

def img2cam(X,cam_intr):
    return X@cam_intr.inverse().transpose(-1,-2)

def cam2world(X,pose):
    # pose is c2w
    X_hom = to_hom(X)
    # pose_inv = invert(pose)
    return X_hom@pose.transpose(-1,-2)

def merge_wireframes(wireframe):
    wireframe_merged = [[],[],[]]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:,0]]+[None]
        wireframe_merged[1] += [float(n) for n in w[:,1]]+[None]
        wireframe_merged[2] += [float(n) for n in w[:,2]]+[None]
    return wireframe_merged

def get_camera_mesh(pose,depth=1):
    vertices = torch.tensor([[-0.5,-0.5,-1],
                             [0.5,-0.5,-1],
                             [0.5,0.5,-1],
                             [-0.5,0.5,-1],
                             [0,0,0]])*depth
    faces = torch.tensor([[0,1,2],
                          [0,2,3],
                          [0,1,4],
                          [1,2,4],
                          [2,3,4],
                          [3,0,4]])
    vertices = cam2world(vertices[None],pose)
    # vertices = vertices[None]
    wireframe = vertices[:,[0,1,2,3,0,4,1,2,4,3]]
    return vertices, faces, wireframe

def merge_meshes(vertices,faces):
    mesh_N,vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)],dim=0)
    vertices_merged = vertices.view(-1,vertices.shape[-1])
    return vertices_merged,faces_merged    

def merge_centers(centers):
    center_merged = [[],[],[]]
    for c1,c2 in zip(*centers):
        center_merged[0] += [float(c1[0]),float(c2[0]),None]
        center_merged[1] += [float(c1[1]),float(c2[1]),None]
        center_merged[2] += [float(c1[2]),float(c2[2]),None]
    return center_merged

@torch.no_grad()
def vis_cameras(opt,vis,step,poses=[],colors=["blue","magenta"],plot_dist=True):
    win_name = "{}/{}".format(opt.group,opt.name)
    data = []
    # set up plots
    centers = []
    for pose,color in zip(poses,colors):
        pose = pose.detach().cpu()
        vertices,faces,wireframe = get_camera_mesh(pose,depth=opt.visdom.cam_depth)
        center = vertices[:,-1]
        centers.append(center)
        # camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:,0]],
            y=[float(n) for n in center[:,1]],
            z=[float(n) for n in center[:,2]],
            mode="markers",
            marker=dict(color=color,size=3),
        ))
        # colored camera mesh
        vertices_merged,faces_merged = merge_meshes(vertices,faces)
        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:,0]],
            y=[float(n) for n in vertices_merged[:,1]],
            z=[float(n) for n in vertices_merged[:,2]],
            i=[int(n) for n in faces_merged[:,0]],
            j=[int(n) for n in faces_merged[:,1]],
            k=[int(n) for n in faces_merged[:,2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))
        # camera wireframe
        wireframe_merged = merge_wireframes(wireframe)
        data.append(dict(
            type="scatter3d",
            x=wireframe_merged[0],
            y=wireframe_merged[1],
            z=wireframe_merged[2],
            mode="lines",
            line=dict(color=color,),
            opacity=0.3,
        ))
    if plot_dist:
        # distance between two poses (camera centers)
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red",width=4,),
        ))
        if len(centers)==4:
            center_merged = merge_centers(centers[2:4])
            data.append(dict(
                type="scatter3d",
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode="lines",
                line=dict(color="red",width=4,),
            ))
    # send data to visdom
    vis._send(dict(
        data=data,
        win="poses",
        eid=win_name,
        layout=dict(
            title="({})".format(step),
            autosize=True,
            margin=dict(l=30,r=30,b=30,t=30,),
            showlegend=False,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
        opts=dict(title="{} poses ({})".format(win_name,step),),
    ))   

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def setup_3D_plot(ax,elev,azim,lim=None):
    ax.xaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.yaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.zaxis.set_pane_color((1.0,1.0,1.0,0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9,0.9,0.9,1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X",fontsize=16)
    ax.set_ylabel("Y",fontsize=16)
    ax.set_zlabel("Z",fontsize=16)
    ax.set_xlim(lim.x[0],lim.x[1])
    ax.set_ylim(lim.y[0],lim.y[1])
    ax.set_zlim(lim.z[0],lim.z[1])
    ax.view_init(elev=elev,azim=azim)

def vis(pose, pose_ref):
    cam_depth = 0.5
    _, _, cam = get_camera_mesh(pose, depth=cam_depth)
    _, _, cam_ref = get_camera_mesh(pose_ref, depth=cam_depth)
    
    color = (0,0.6,0.7)
    ref_color = (0.7,0.2,0.7)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    setup_3D_plot(ax,elev=45,azim=35,lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4)))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    
    N = len(cam)
    
    ################################ poses 1 #################################
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=color))
    for i in range(N):
        ax.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=color,linewidth=1)
        ax.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=color,s=20)
    ################################ poses 2 #################################
    ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
    for i in range(N):
        ax.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=ref_color,linewidth=0.5)
        ax.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)
    ################################ translation error #################################    
    for i in range(N):
        ax.plot([cam[i,5,0],cam_ref[i,5,0]],
                [cam[i,5,1],cam_ref[i,5,1]],
                [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
    png_fname = "xxx.png"
    plt.savefig(png_fname,dpi=75)
    # 显示图形
    plt.show()
from torch.utils.data import DataLoader
if __name__ == '__main__':
    dataset = BlenderDataSet()
    # print(dataset.raw_poses[0])
    vis(dataset.raw_poses[:, :3, :], dataset.noise_poses[:, :3, :])
    # dataloader = DataLoader(dataset)
    
    # To save image:
    # i1 = self.images[0] * 255
    # imageio.v2.imwrite('test2.png', i1.astype(np.uint8))        