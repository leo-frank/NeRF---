import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from utils.log import logger


class VanillaNeRF(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D, small):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(VanillaNeRF, self).__init__()

        logger.title("VanillaNeRF, small:{}".format(small))
        logger.info("position in_dims: {}, level: {}".format(pos_in_dims, (pos_in_dims-3)//6))
        logger.info("direction in_dims: {}, level: {}".format(dir_in_dims, (dir_in_dims-3)//6))
        logger.info("hidden dimension: {}".format(D))
        
        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims

        if not small:
            self.layers0 = nn.Sequential(
                    nn.Linear(pos_in_dims, D), nn.ReLU(),
                    nn.Linear(D, D), nn.ReLU(),
                    nn.Linear(D, D), nn.ReLU(),
                    nn.Linear(D, D), nn.ReLU(),
            )

            self.layers1 = nn.Sequential(
                    nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut
                    nn.Linear(D, D), nn.ReLU(),
                    nn.Linear(D, D), nn.ReLU(),
                    nn.Linear(D, D), nn.ReLU(),
            )
        else:
            self.layers0 = nn.Sequential(
                    nn.Linear(pos_in_dims, D), nn.ReLU(),
                    nn.Linear(D, D), nn.ReLU(),
                    nn.Linear(D, D), nn.ReLU(),
            )

            self.layers1 = nn.Sequential(
                    nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut
                    nn.Linear(D, D), nn.ReLU(),
            )
            

        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_rgb = nn.Linear(D//2, 3)
        self.rgb_activation = nn.Sigmoid()
        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_rgb.bias.data = torch.tensor([0.02, 0.02, 0.02]).float()

    def forward(self, pos_enc, dir_enc):
        """
        :param pos_enc: (B, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (B, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (B, N_sample, 4)
        """
        x = self.layers0(pos_enc)  # (B, N_sample, D)
        x = torch.cat([x, pos_enc], dim=2)  # (B, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (B, N_sample, D)

        density = self.fc_density(x)  # (B, N_sample, 1)

        feat = self.fc_feature(x)  # (B, N_sample, D)
        x = torch.cat([feat, dir_enc], dim=2)  # (B, N_sample, D+dir_in_dims)
        x = self.rgb_layers(x)  # (B, N_sample, D/2)
        rgb = self.rgb_activation(self.fc_rgb(x))  # (B, N_sample, 3)

        return rgb, density