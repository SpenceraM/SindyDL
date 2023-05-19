import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import AutoEncoder

# class Trainer:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = AutoEncoder(cfg).to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['learning_rate'])
#         self.criterion = nn.MSELoss()

def train(train_data, val_dat, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # not used for debugging
    model = AutoEncoder(cfg)#.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = nn.MSELoss()