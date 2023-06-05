import torch
import torch.nn as nn

class Renderer(nn.Module):
    
    def __init__(self):
        pass
    
    @staticmethod
    def volume_rendering(sigma, color, t):
        """
        :param sigma: (N_rays, N_samples, 1)
        :param color: (N_rays, N_samples, 3)
        :param t: (N_rays, N_samples, 1)
        """
        sigma = torch.relu(sigma) # (N_rays, N_samples, 1)
        N_rays, N_samples, _ = sigma.shape
        delta = t[:, 1:, ] - t[:, :-1, ] # (N_rays, N_samples-1, 1) # NOTE: ensure rays_d has been normalized
        delta_far = torch.empty(size=(N_rays, 1, 1), dtype=torch.float32, device=delta.device).fill_(1e10) # (N_rays, 1, 1)
        delta = torch.cat([delta, delta_far], 1) # (N_rays, N_samples, 1)
        alpha = 1 - torch.exp(-1.0 * sigma * delta) # (N_rays, N_samples, 1)
        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :1, :]), 1-alpha], 1) # # (N_rays, N_samples + 1, 1)
        alpha_shifted = alpha_shifted.squeeze(2) # # (N_rays, N_samples + 1)
        transmittance =  torch.cumprod(alpha_shifted, dim=-1)[:, :-1].unsqueeze(2)  # (N_rays, N_samples, 1) # TODO: 1-alphas+1e-10 ???
        weight = alpha * transmittance # (N_rays, N_samples, 1)
        rgb = torch.sum(weight * color, dim=1) # (N_rays, 3)
        depth = torch.sum(weight * t, dim=1) # (N_rays, 1)
        return rgb, depth



    
    