import time
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from vanilla_nerf import VanillaNeRF
from hash_nerf import HashNeRF
from embedding import Embedder
from hash_embedding import HashEmbedder, SHEncoder
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


### dataloading: ray_o, rays_d, near, far ###
batch_size = cfg.train.batch_size if mode == 'train' else cfg.test.batch_size
shuffle = cfg.train.shuffle  if mode == 'train' else cfg.test.shuffle
chunk_size = cfg.train.chunk_size if mode == 'train' else cfg.test.chunk_size
inference_train =  cfg.test.inference_train  if mode == 'test' else False
train_dataset = BlenderDataSet(cfg.data.base_dir, cfg.data.scene, mode='train', inference_train=False) # TODO: clecan codes
test_dataset = BlenderDataSet(cfg.data.base_dir, cfg.data.scene, mode='test', inference_train=inference_train)
dataset = BlenderDataSet(cfg.data.base_dir, cfg.data.scene, mode=mode, inference_train=inference_train)
h, w = dataset.h, dataset.w
dataloader = DataLoader(dataset, batch_size, shuffle)


### xyz embedding & direction embedding ###
if cfg.model.type == 'vanilla':
    embedder_xyz_model = Embedder(input_dim = 3, level = cfg.model.Embedding.xyz_level, description = "Position Embedder").to(device)
    embedder_direction_model = Embedder(input_dim = 3, level = cfg.model.Embedding.direction_level, description = "Direction Embedder").to(device)
elif cfg.model.type == 'hash':
    embedder_xyz_model = HashEmbedder(bounding_box = train_dataset.bounding_box, description = "Multi-level Hash Position Embedder").to(device)
    embedder_direction_model = SHEncoder(description = "SH Direction Embedder").to(device)

### nerf model ###
hidden_dim = cfg.model.VanillaNeRF.hidden_size if cfg.model.type == 'vanilla' else cfg.model.HashNeRF.hidden_size
small = False if cfg.model.type == 'vanilla' else True
if cfg.model.type == 'vanilla':
    nerf_model = VanillaNeRF(pos_in_dims=embedder_xyz_model.out_dim, dir_in_dims=embedder_direction_model.out_dim, D=hidden_dim, small=small).to(device)
elif cfg.model.type == 'hash':
    nerf_model = HashNeRF(input_ch=embedder_xyz_model.out_dim, input_ch_views=embedder_direction_model.out_dim).to(device)
### loss model ###
loss_model = torch.nn.MSELoss(reduction='mean').to(device)

### optimizer ###
if cfg.model.type == 'vanilla':
    optimizer = optim.Adam(nerf_model.parameters(), lr=cfg.lr.initial)
elif cfg.model.type == 'hash':
    optimizer = optim.Adam(nerf_model.parameters(), lr=1e-2, eps=1e-15)
    optimizer_embedding_xyz = optim.Adam(embedder_xyz_model.parameters(), lr=1e-2, eps=1e-15)

### prepare output dir ###
log_dir = './output/{}/{}/logs/'.format(cfg.data.scene, cfg.description)
eval_dir = './output/{}/{}/test/'.format(cfg.data.scene, cfg.description) if not inference_train else './output/{}/{}/test_trained/'.format(cfg.data.scene, cfg.description)
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
    
    progress_bar = tqdm(total=max_epoch*len(dataloader), desc='Training', unit='batch')
    

elif mode == 'test':
    
    torch.set_grad_enabled(False)
    
    embedder_xyz_model.eval()
    embedder_direction_model.eval()
    nerf_model.eval()
    
    # state_dict = torch.load(cfg.test.ckpt)
    # nerf_model.load_state_dict(state_dict)

    checkpoint = torch.load(cfg.test.ckpt)
    if cfg.old:
        nerf_model.load_state_dict(checkpoint)
    else:
        nerf_model.load_state_dict(checkpoint['nerf_model_state_dict'])
        embedder_xyz_model.load_state_dict(checkpoint['embedder_xyz_model_state_dict'])
    progress_bar = tqdm(total=max_epoch*len(dataloader), desc='Inference', unit='batch')
    
    

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
        xyz_embedded = embedder_xyz_model(xyz_samples)                              # (N_rays, N_samples, xxx)
        direction_embedded = embedder_direction_model(rays_d)                       # (N_rays, N_samples, yyy)
        color, sigma = nerf_model(xyz_embedded, direction_embedded)
        rgb_prediction, depth_prediction = Renderer.volume_rendering(sigma, color, z_vals, cfg.data.type) # (N_rays, 3), (N_rays, 1)
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

def forward_with_no_gradients(batch, cfg):
    with torch.no_grad():
        return forward(batch, cfg)
global_step = 0    
all_psnr = []
for epoch in range(max_epoch): # max_epoch is 1 for test
    for it, batch in enumerate(dataloader):
        
        rgb_prediction, rgbs = forward(batch, cfg)  # (B, H*W, 3)
        
        if mode == 'train':
            rgb_loss = loss_model(rgb_prediction, rgbs)
            psnr = compute_psnr(rgb_prediction, rgbs)
             
            optimizer.zero_grad()
            if cfg.model.type == 'hash':
                optimizer_embedding_xyz.zero_grad()

            rgb_loss.backward()

            optimizer.step()
            if cfg.model.type == 'hash':
                optimizer_embedding_xyz.step()

            writer.add_scalar('loss/rgb', rgb_loss.item(), global_step)
            writer.add_scalar('loss/psnr', psnr.item(), global_step)
           
        else: # evaluation mode
            rgb_prediction = rgb_prediction.unbind(0)
            rgbs = rgbs.unbind(0)
            
            for i, (img, img_gt) in enumerate(zip(rgb_prediction, rgbs)): # in practice, i always equals to 0
                img = img.view(h, w, 3) 
                img_gt = img_gt.view(h, w, 3).to(device)
                concatenated_image = torch.cat([img, img_gt], dim=1) # (h, 2w, 3)
                save_image(concatenated_image.permute(2, 0, 1), "{}/{}_{}.png".format(eval_dir, it, i))
                psnr = compute_psnr(img.unsqueeze(0), img_gt.unsqueeze(0))
                all_psnr.append(psnr)
                if inference_train == False:
                    writer.add_scalar('inference/psnr', psnr.item(), it)
                else:
                    writer.add_scalar('inference_train/psnr', psnr.item(), it)
                logger.info("saving {}_{}.png, psnr: {}".format(it, i, psnr))
            
        global_step += 1
        progress_bar.update(1)

        if mode == 'train':
            progress_bar.set_postfix({'Epoch': epoch, 'Loss': rgb_loss.item(), 'PSNR': psnr.item()})
        else:
            progress_bar.set_postfix({'Epoch': epoch, 'PSNR': psnr.item()})
        
        if global_step % 1000 == 0:
            if cfg.model.type == 'hash':
                checkpoint = {
                    'nerf_model_state_dict': nerf_model.state_dict(),
                    'embedder_xyz_model_state_dict': embedder_xyz_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            else:
                checkpoint = {
                    'nerf_model_state_dict': nerf_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                
            torch.save(checkpoint, '{}/model{}_{}.pth'.format(ckpt_dir, epoch, global_step)) 
        
        if global_step % 500 == 1:
            x1, x2, x3 = test_dataset[0]['rays_o'].unsqueeze(0), test_dataset[0]['rays_d'].unsqueeze(0), test_dataset[0]['rgbs'].unsqueeze(0)
            test_batch = {
                'rays_o': x1, # (B, 3)
                'rays_d': x2, # (B, 3)
                'rgbs': x3, # (B, 3)
            }
            rgb_prediction, rgbs = forward_with_no_gradients(test_batch, cfg)
            for i, (img, img_gt) in enumerate(zip(rgb_prediction, rgbs)):
                img = img.view(h, w, 3) 
                img_gt = img_gt.view(h, w, 3).to(device)
                concatenated_image = torch.cat([img, img_gt], dim=1) # (h, 2w, 3)    
                save_image(concatenated_image.permute(2, 0, 1), "{}/test_{}_{}.png".format(eval_dir, global_step, i))  
                psnr = compute_psnr(img.unsqueeze(0), img_gt.unsqueeze(0))
                all_psnr.append(psnr)
                logger.info("saving {}/test_{}_{}.png, psnr: {}".format(eval_dir, global_step, it, psnr))
                writer.add_scalar('train_val/psnr', psnr.item(), global_step)  
        
    if mode == 'train':
        pass
    else: # evaluation mode
        logger.title("average psnr: {}".format(torch.tensor(all_psnr).mean().item()))
        generate_video_from_images(eval_dir)
        