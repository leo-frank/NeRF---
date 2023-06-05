import torch
import torch.nn as nn
from utils.logging import logger

class Embedder(nn.Module):
    def __init__(self, input_dim, level, description):
        """
        :param input_dim: int
        :param level: int
        """
        super(Embedder, self).__init__()
        self.out_dim = input_dim * 2 * level + input_dim
        
        logger.title(description)
        logger.info("input dimension: {}".format(input_dim))
        logger.info("encoding level: {}".format(level))
        
        self.funcs = [torch.sin, torch.cos] #
        self.freqs = [2**i for i in range(level)]
        
        logger.info("output dimension: {}".format(self.out_dim))
    
    def forward(self, x):
        """
        :param x:   (B, input_dim)
        :return: (B, output_dim)
        """
        res = [x]
        for func in self.funcs:
            for freq in self.freqs:
                res.append(func(freq*x))
        return torch.cat(res, -1)
            
                
        
        