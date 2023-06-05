import time
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from vanilla_nerf import VanillaNeRF
from embedding import Embedder
from rendering import Renderer
from blender import BlenderDataSet
from utils.logging import logger
from utils.metrics import compute_psnr
import matplotlib.pyplot as plt

mode = 'train'
mode = 'test'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### xyz embedding ###
xyz_level = 10
embedder_xyz_model = Embedder(input_dim = 3, level = xyz_level, description = "Position Embedder").to(device)

### direction embedding ###
direction_level = 4
embedder_direction_model = Embedder(input_dim = 3, level = direction_level, description = "Direction Embedder").to(device)

### nerf model ###
nerf_model = VanillaNeRF(embedder_xyz_model.out_dim, embedder_direction_model.out_dim, 128).to(device)

### loss model ###
loss_model = torch.nn.MSELoss(reduction='mean').to(device)

### optimizer ###
initial_lr = 0.001
optimizer = optim.Adam(nerf_model.parameters(), lr=initial_lr)


### dataloading: ray_o, rays_d, near, far ###
batch_size = 10240 if mode == 'train' else 1
shuffle = True if mode == 'train' else False
dataset = BlenderDataSet(mode=mode)
h, w = dataset.h, dataset.w
dataloader = DataLoader(dataset, batch_size, shuffle)


### tensorboard ###
writer = SummaryWriter()
global_step = 0

logger.title("Current mode: {}".format(mode))

if mode == 'train':
    
    embedder_xyz_model.train()
    embedder_direction_model.train()
    nerf_model.train()

    for epoch in range(30):
        for it, batch in enumerate(dataloader):
            logger.info("epoch: {}, iter: {}/{}".format(epoch, it, len(dataloader)))
            rays_o, rays_d, rgbs = batch['rays_o'].to(device), batch['rays_d'].to(device), batch['rgbs'].to(device) # (B, 3), (B, 3), (B, 3)
            near = 0
            far = 10

            ### samples on o+td ### # TODO: perturb
            N_samples = 256
            rays_o = rays_o.unsqueeze(1).expand(batch_size, N_samples, 3) # (B, N_samples, 3)
            rays_d = rays_d.unsqueeze(1).expand(batch_size, N_samples, 3) # (B, N_samples, 3)
            z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device) # (N_samples)
            z_vals = near * (1-z_steps) + far * z_steps # (N_samples)
            z_vals = z_vals.unsqueeze(0).unsqueeze(2).expand(batch_size, N_samples, 1) # (B, N_samples, 3)
            xyz_samples = rays_o + rays_d * z_vals # (B, N_samples, 3)

            ### model inference ###
            xyz_embedded = embedder_xyz_model(xyz_samples)
            direction_embedded = embedder_direction_model(rays_d)
            color, sigma = nerf_model(xyz_embedded, direction_embedded)
            rgb_prediction, depth_prediction = Renderer.volume_rendering(sigma, color, z_vals) # (N_rays, 3), (N_rays, 1)

            rgb_loss = loss_model(rgb_prediction, rgbs)
            
            optimizer.zero_grad()
            rgb_loss.backward()
            optimizer.step()
            
            writer.add_scalar('Loss', rgb_loss.item(), global_step)
            global_step += 1

        torch.save(nerf_model.state_dict(), 'output/ckpts/model_state_{}.pth'.format(epoch))

elif mode == 'test':
    
    embedder_xyz_model.eval()
    embedder_direction_model.eval()
    nerf_model.eval()
    
    load_epoch = 29
    state_dict = torch.load('output/ckpts/model_state_{}.pth'.format(load_epoch))
    nerf_model.load_state_dict(state_dict)
    
    all_psnr = []

    with torch.no_grad():
        for it, batch in enumerate(dataloader):
            logger.info("epoch: {}, iter: {}/{}".format(0, it, len(dataloader)))
            # NOTE: WHEN TO 
            rays_o, rays_d, rgbs = batch['rays_o'], batch['rays_d'], batch['rgbs'] # (B, H*W, 3), # (B, H*W, 3), # (B, H*W, 3)

            shape = rays_o.shape
            
            
            rays_o = rays_o.view(-1, 3)
            rays_d = rays_d.view(-1, 3)
            
            near = 0
            far = 10

            ### samples on o+td ### # TODO: perturb
            N_samples = 256
            rays_o = rays_o.unsqueeze(1).expand(-1, N_samples, 3) # (B, N_samples, 3)
            rays_d = rays_d.unsqueeze(1).expand(-1, N_samples, 3) # (B, N_samples, 3)
            z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device) # (N_samples)
            z_vals = near * (1-z_steps) + far * z_steps # (N_samples)
            z_vals = z_vals.unsqueeze(0).unsqueeze(2).expand(rays_o.shape[0], N_samples, 1) # (B, N_samples, 1)
            xyz_samples = rays_o + rays_d * z_vals # (B, N_samples, 3)
            logger.info("xyz_samples.shape = {}".format(xyz_samples.shape))
            logger.info("rays_d.shape = {}".format(rays_d.shape))
            logger.info("z_vals.shape = {}".format(z_vals.shape))

            def inference_chunk(xyz_samples, rays_d, z_vals):
                ### model inference ###
                xyz_embedded = embedder_xyz_model(xyz_samples)
                direction_embedded = embedder_direction_model(rays_d)
                color, sigma = nerf_model(xyz_embedded, direction_embedded)
                rgb_prediction, depth_prediction = Renderer.volume_rendering(sigma, color, z_vals) # (N_rays, 3), (N_rays, 1)
                return rgb_prediction, depth_prediction

            results =[]
            chunk_size = 40960
            start_time = time.time()
            
            for i in range(0, xyz_samples.shape[0], chunk_size):
                rgb, depth = inference_chunk(xyz_samples[i:i+chunk_size].to(device), rays_d[i:i+chunk_size].to(device), z_vals[i:i+chunk_size].to(device))
                results.append(rgb)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time:", elapsed_time, "seconds")
            rgb_prediction = torch.cat(results)
            rgb_prediction = rgb_prediction.view(shape) # (B, H*W, 3)
            
            rgb_prediction = rgb_prediction.unbind(0)
            rgbs = rgbs.unbind(0)
            
            for i, (img, img_gt) in enumerate(zip(rgb_prediction, rgbs)):
                img = img.view(h, w, 3) 
                img_gt = img_gt.view(h, w, 3).to(device)
                concatenated_image = torch.cat([img, img_gt], dim=1) # (h, 2w, 3)
                save_image(concatenated_image.permute(2, 0, 1), "{}_{}.png".format(it, i))
                psnr = compute_psnr(img.unsqueeze(0), img_gt.unsqueeze(0))
                all_psnr.append(psnr)
                logger.info("saving {}_{}.png, psnr: {}".format(it, i, psnr))
        logger.title("average psnr: {}".format(torch.tensor(all_psnr).mean().item()))