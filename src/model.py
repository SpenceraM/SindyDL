import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.input_dim  = cfg['input_dim']
        self.latent_dim = cfg['latent_dim']
        self.activation = cfg['activation']
        self.poly_order = cfg['poly_order']
        if 'include_sine' in cfg.keys():
            self.include_sine = cfg['include_sine']
        else:
            self.include_sine = False
        self.library_dim = cfg['library_dim']
        self.model_order = cfg['model_order']
        self.sequential_thresholding = cfg['sequential_thresholding']
        
        # TODO
        # Not sure if i need tensors here for x, dx, ddx or if they will be passed in forward function

        if self.activation == 'linear':
            pass
        else:  # non-linear activation
            self.encoder = XcoderHalf(self.input_dim, self.latent_dim, [], self.activation)
            self.decoder = XcoderHalf(self.latent_dim, self.input_dim, [], self.activation)

        if self.model_order == 1:
            self.z_derivative_func  = ZDerivativeOrder1(self.activation)
            self.sindy_library = SINDyLibraryOrder1(self.poly_order, self.include_sine)
        else:
            raise ValueError('Invalid model order')
            # self.z_derivative_func  =
            # self.sindy_library =
        if self.c
    def forward(self, x):
        z = self.encoder(x)


        x_hat = self.decoder(z)


class XcoderHalf(nn.Module):  # Xcoder as in Encoder or Decoder
    def __init__(self, input_dim, output_dim, widths, activation):
        super().__init__()
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        # self.activation = activation

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'elu':
            activation = nn.ELU()
        else:
            raise ValueError('Invalid activation function')

        self.layers = nn.ModuleList()
        for i in range(len(widths)):
            if i == 0:
                self.layers.append(FcLayer(input_dim, widths[i], activation))
            else:
                self.layers.append(FcLayer(widths[i-1], widths[i], activation))
        self.layers.append(FcLayer(widths[-1], output_dim, activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FcLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class ZDerivativeOrder1(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x, dx, encoder):
        dz = dx
        if self.activation == 'elu':
            for i in range(len(encoder.layers)-1):
                x = encoder.layers[i](x)
                dz = torch.multiply(torch.min(torch.exp(x), torch.ones_like(x)), torch.matmul(dz,encoder.layers[i].fc.weight))
                x = nn.ELU(x)
        if self.activation == 'relu':
            for i in range(len(encoder.layers)-1):
                x = encoder.layers[i](x)
                dz = torch.multiply(torch.max(torch.sign(x),0), torch.matmul(dz,encoder.layers[i].fc.weight))
                x = nn.ReLU(x)
        if self.activation == 'sigmoid':
            for i in range(len(encoder.layers)-1):
                x = encoder.layers[i](x)
                x = nn.Sigmoid(x)
                dz = torch.multiply(torch.multiply(x,1-x), torch.matmul(dz,encoder.layers[i].fc.weight))
        else:
            for i in range(len(encoder.layers)-1):
                dz = torch.matmul(dz,encoder.layers[i].fc.weight)
        dz = torch.matmul(dz,encoder.layers[-1].fc.weight)
        return dz


class SINDyLibraryOrder1(nn.Module):
    def __init__(self,  poly_order, include_sine):
        super().__init__()
        self.poly_order = poly_order
        self.include_sine = include_sine
    def forward(self, z):
        n_times = z.shape[0]
        latent_dim = z.shape[1]

        library = torch.ones(n_times, 1)
        library = torch.cat((library, z), 1) # order = 1

        if self.poly_order > 1: # order = 2
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    library = torch.cat((library, z[:, i]*z[:, j]), 1)

        if self.poly_order > 2:  # order = 3
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        library = torch.cat((library,z[:, i] * z[:, j] * z[:, k]),1)

        if self.poly_order > 3: # order = 4
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        for p in range(k, latent_dim):
                            library = torch.cat((library,z[:, i] * z[:, j] * z[:, k] * z[:, p]),1)

        if self.poly_order == 5: # order = 5
            for i in range(latent_dim):
                for j in range(i, latent_dim):
                    for k in range(j, latent_dim):
                        for p in range(k, latent_dim):
                            for q in range(p, latent_dim):
                                library = torch.cat((library, z[:, i] * z[:, j] * z[:, k] * z[:, p] * z[:, q]),1)
        if self.poly_order > 5:
            raise ValueError('poly_order > 5 not implemented')

        if self.include_sine:
            library = torch.cat((library,torch.sin(z)),1)

        return library