import time
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from vanilla_nerf import VanillaNeRF
from embedding import Embedder
from rendering import Renderer
from blender import BlenderDataSet
from utils.log import logger
from utils.metrics import compute_psnr
from utils.vis_utils import generate_video_from_images
from config_parse import config 
import matplotlib.pyplot as plt

### config ###
cfg = config()
mode = cfg.mode
assert mode in ['train', 'test']
max_epoch = cfg.train.max_epoch if mode == 'train' else 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### xyz embedding ###
embedder_xyz_model = Embedder(input_dim = 3, level = cfg.model.Embedding.xyz_level, description = "Position Embedder").to(device)

### direction embedding ###
embedder_direction_model = Embedder(input_dim = 3, level = cfg.model.Embedding.direction_level, description = "Direction Embedder").to(device)

### nerf model ###
nerf_model = VanillaNeRF(embedder_xyz_model.out_dim, embedder_direction_model.out_dim, cfg.model.VanillaNeRF.hidden_size).to(device)

### loss model ###
loss_model = torch.nn.MSELoss(reduction='mean').to(device)

### optimizer ###
optimizer = optim.Adam(nerf_model.parameters(), lr=cfg.lr.initial)


### dataloading: ray_o, rays_d, near, far ###
batch_size = cfg.train.batch_size if mode == 'train' else cfg.test.batch_size
shuffle = cfg.train.shuffle  if mode == 'train' else cfg.test.shuffle
chunk_size = cfg.train.chunk_size if mode == 'train' else cfg.test.chunk_size
dataset = BlenderDataSet(cfg.data.base_dir, cfg.data.scene, mode=mode)
h, w = dataset.h, dataset.w
dataloader = DataLoader(dataset, batch_size, shuffle)

### prepare output dir ###
log_dir = './output/{}/{}/logs/'.format(cfg.data.scene, cfg.description)
eval_dir = './output/{}/{}/test/'.format(cfg.data.scene, cfg.description)
ckpt_dir = './output/{}/{}/ckpt/'.format(cfg.data.scene, cfg.description)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

### tensorboard ###
writer = SummaryWriter(log_dir)
global_step = 0

logger.title("Current mode: {}".format(mode))


if mode == 'train':
    
    torch.set_grad_enabled(True)
    
    embedder_xyz_model.train()
    embedder_direction_model.train()
    nerf_model.train()
    

elif mode == 'test':
    
    torch.set_grad_enabled(False)
    
    embedder_xyz_model.eval()
    embedder_direction_model.eval()
    nerf_model.eval()
    
    state_dict = torch.load(cfg.test.ckpt)
    nerf_model.load_state_dict(state_dict)
    
    

def forward(batch, cfg):
    rays_o, rays_d, rgbs = batch['rays_o'], batch['rays_d'], batch['rgbs'] # (B, H*W, 3), # (B, H*W, 3), # (B, H*W, 3)
    shape = rays_o.shape
    
    rays_o = rays_o.view(-1, 3)                                                     # (B*H*W, 3)
    rays_d = rays_d.view(-1, 3)                                                     # (B*H*W, 3)
                                                                                    
     # Next, B*H*W is called N_rays
     
    near = 2
    far = 6

    ### samples on o+td ### # TODO: perturb
    N_samples = cfg.N_samples
    rays_o = rays_o.unsqueeze(1).expand(-1, N_samples, 3)                           # (N_rays, N_samples, 3)
    rays_d = rays_d.unsqueeze(1).expand(-1, N_samples, 3)                           # (N_rays, N_samples, 3)
    z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device)                 # (N_samples)
    z_vals = near * (1-z_steps) + far * z_steps                                     # (N_samples)
    z_vals = z_vals.unsqueeze(0).unsqueeze(2).expand(rays_o.shape[0], N_samples, 1) # (N_rays, N_samples, 1)
    xyz_samples = rays_o + rays_d * z_vals                                          # (N_rays, N_samples, 3)

    def inference_chunk(xyz_samples, rays_d, z_vals):
        ### model inference ###
        xyz_embedded = embedder_xyz_model(xyz_samples)
        direction_embedded = embedder_direction_model(rays_d)
        color, sigma = nerf_model(xyz_embedded, direction_embedded)
        rgb_prediction, depth_prediction = Renderer.volume_rendering(sigma, color, z_vals) # (N_rays, 3), (N_rays, 1)
        return rgb_prediction, depth_prediction

    results =[]
    
    ### TODO: This is not great implementation. I want to reuse codes as possible, but chunk_size is not really suited for training.
    ### Given same batch input, forward_gradient and forward_no_gradient indded use different amount of mem, required for gradient bookkeeping.
    for i in range(0, xyz_samples.shape[0], chunk_size):
        rgb, depth = inference_chunk(xyz_samples[i:i+chunk_size].to(device), rays_d[i:i+chunk_size].to(device), z_vals[i:i+chunk_size].to(device))
        results.append(rgb)

    rgb_prediction = torch.cat(results)
    rgb_prediction = rgb_prediction.view(shape) # (B, H*W, 3)
    

    return rgb_prediction, rgbs.to(rgb_prediction.device)

global_step = 0    
all_psnr = []
for epoch in range(max_epoch): # max_epoch is 1 for test
    for it, batch in enumerate(dataloader):
        
        start_time = time.time()
        
        logger.info("epoch: {}, iter: {}/{}".format(epoch, it, len(dataloader)))
        rgb_prediction, rgbs = forward(batch, cfg)  # (B, H*W, 3)
        
        if mode == 'train':
            rgb_loss = loss_model(rgb_prediction, rgbs)
            
            optimizer.zero_grad()
            rgb_loss.backward()
            optimizer.step()
            writer.add_scalar('loss/rgb', rgb_loss.item(), global_step)
            
        else: # evaluation mode
            rgb_prediction = rgb_prediction.unbind(0)
            rgbs = rgbs.unbind(0)
            
            for i, (img, img_gt) in enumerate(zip(rgb_prediction, rgbs)):
                img = img.view(h, w, 3) 
                img_gt = img_gt.view(h, w, 3).to(device)
                concatenated_image = torch.cat([img, img_gt], dim=1) # (h, 2w, 3)
                save_image(concatenated_image.permute(2, 0, 1), "{}/{}_{}.png".format(eval_dir, it, i))
                psnr = compute_psnr(img.unsqueeze(0), img_gt.unsqueeze(0))
                all_psnr.append(psnr)
                logger.info("saving {}_{}.png, psnr: {}".format(it, i, psnr))
            
        global_step += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("Elapsed time: {} seconds".format(elapsed_time))
        
    if mode == 'train':
        torch.save(nerf_model.state_dict(), '{}/model_state_{}.pth'.format(ckpt_dir, epoch))    
    else: # evaluation mode
        logger.title("average psnr: {}".format(torch.tensor(all_psnr).mean().item()))
        generate_video_from_images(eval_dir)
        